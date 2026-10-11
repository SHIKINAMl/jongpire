"""
CPU の席を動かす

人間のクライアントと同じ通知（GameSession が CPU 宛てに送る payload）を receive で受け取り、
そのたびにエンジンの状態から「いま自分がやるべきこと」を調べて行動する。
行動は人間と同じ action JSON を GameSession.handle_game_action に渡すだけなので、
検証・状態遷移・相手への通知はすべて既存の処理を通る。

- 行動の前に、設定の think_time の範囲からランダムに選んだ秒数だけ待つ（考えている雰囲気を出す）
- 待っている間に状況が変わることがあるので、待ち終わってからもう一度やるべきことを調べ直す
- 行動が弾かれたら（error が返ったら）安全な手で打ち直す。それでも通らなければ、その場面は諦めて次の通知を待つ
"""
import asyncio
import logging
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from ..engine.game_engine import MULLIGAN_CANDIDATE_COUNT
from ..engine.game_state import RoundStatus
from .cpu_config import CpuProfile
from .observation import CpuObservation, PublicInfo, build_observation
from .rule_brain import HandPlan, create_brain, fallback_bet, fallback_discard, fallback_hand_indexes

if TYPE_CHECKING:
    from ..communication.game_session import GameSession

logger = logging.getLogger(__name__)

# 手牌の候補探索（重い）を動かすスレッド。全マッチで共有し、同時に走る探索の数を抑える
# （512 MB のサーバーで、CPU 戦が重なってもスレッドとメモリが増えすぎないように）
_SEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu-search")


class CpuController:
    """1つの CPU 席。マッチの開始から破棄まで生きる。"""

    # 同じ場面で行動を試す回数（1回目は考え方どおり、2回目以降は安全な手）
    MAX_ATTEMPTS_PER_STEP = 3
    # 手牌選択フェーズで続けて使うスキルの上限（考え方の不具合で回り続けないように）
    MAX_SKILL_STEPS = 6

    def __init__(
        self,
        session: "GameSession",
        match_id: str,
        client_id: str,
        profile: CpuProfile,
        rng: Optional[random.Random] = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ):
        self.match_id = match_id
        self.client_id = client_id
        self.profile = profile
        self._session = session
        self._rng = rng or random.Random()
        self._sleep = sleep
        self._brain = create_brain(profile.config, self._rng)

        self._public = PublicInfo()
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._last_error: Optional[str] = None
        self._pending_confirm: Optional[list[int]] = None
        # マリガンの候補（skill_accepted の mulliganOptions。本人にだけ届く）
        self._mulligan_options: Optional[dict] = None
        self._plan: Optional[HandPlan] = None
        self._attempts: Counter = Counter()

    # ========== 通知の受け取り ==========

    def receive(self, payload: dict[str, Any]) -> None:
        """GameSession から CPU 宛ての通知を受け取る（人間なら WebSocket で届くもの）。"""
        if self._stopped:
            return

        msg_type = payload.get("type")
        data = payload.get("data")
        if not isinstance(data, dict):
            data = {}

        if msg_type == "round_start":
            self._public.reset_round()
            self._plan = None
            self._pending_confirm = None
            self._attempts.clear()
        elif msg_type == "opening_boost_assigned":
            for boost in data.get("boosts", []) or []:
                if boost.get("client_id") != self.client_id:
                    name = boost.get("yaku_name")
                    self._public.opponent_opening_boosts[name] = (
                        self._public.opponent_opening_boosts.get(name, 0) + int(boost.get("bonus_han", 1))
                    )
        elif msg_type == "skill_casted":
            perspective = data.get("perspectiveResult")
            if data.get("player_id") == self.client_id and isinstance(perspective, dict):
                self._public.record_my_perspective(
                    perspective.get("suit"), len(perspective.get("newlyExposedIndexes") or [])
                )
            if data.get("player_id") != self.client_id and data.get("source", "skill") == "skill":
                skill = str(data.get("skillType"))
                self._public.opponent_skills_this_round.append(skill)
                if skill == "boost_hand":
                    self._public.opponent_boost_use_count += 1
        elif msg_type == "error":
            self._last_error = str(payload.get("message", data.get("message", "")))
        elif msg_type == "hand_selection_confirmation_required":
            # 満貫未満・非聴牌の手牌を select した。そのまま確定させる
            self._pending_confirm = list(data.get("hand_indexes") or [])
        elif msg_type == "hand_selection_accepted":
            self._pending_confirm = None
        elif msg_type == "skill_accepted" and isinstance(data.get("mulliganOptions"), dict):
            self._mulligan_options = data["mulliganOptions"]
        elif msg_type == "mulligan_selected":
            if self._mulligan_options is not None:
                changes = [
                    {"wall_index": c.get("wallIndex"), "old_tile": c.get("oldTile"), "new_tile": c.get("newTile")}
                    for c in data.get("changes") or []
                ]
                self._public.record_mulligan(self._pending_from_options(self._mulligan_options), changes)
            self._mulligan_options = None

        self.wake()

    def stop(self) -> None:
        """マッチの破棄時に呼ぶ。以後は何もしない。"""
        self._stopped = True
        task = self._task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    # ========== 行動の段取り ==========

    def wake(self) -> None:
        """
        やるべきことがあれば動き出す。通知を受け取ったときのほか、
        CPU 宛ての通知がないまま状態が変わったとき（人間のロンが不成立で手番が CPU に残る など）にも呼ぶ。
        """
        if self._stopped:
            return
        if self._task is not None and not self._task.done():
            # 動いている間は、行動のたびにやるべきことを調べ直すので取りこぼさない
            return
        if self._current_need() is None:
            return
        self._task = self._session.spawn(self.match_id, self._run())

    def _current_need(self) -> Optional[tuple[str, tuple]]:
        """
        いま自分がやるべきことを (場面, 場面を区別するキー) で返す。何もなければ None。
        キーは「同じ場面で何回試したか」を数えるのに使う。
        """
        engine = self._session.get_engine(self.match_id)
        if engine is None:
            return None
        me = engine.get_player_by_id(self.client_id)
        if me is None:
            return None

        round_state = engine.state.round_state
        status = round_state.status
        round_number = round_state.round_number

        if status == RoundStatus.HAND_SELECTION:
            if me.pending_mulligan is not None:
                return "mulligan_select", (round_number, "mulligan_select", len(round_state.reserved_tiles))
            if self._pending_confirm is not None:
                return "select_confirm", (round_number, "select_confirm")
            if not self._session.is_hand_selection_confirmed(self.match_id, self.client_id):
                return "hand_selection", (round_number, "hand_selection")
        elif status == RoundStatus.BETTING:
            if me.bet <= 0:
                return "bet", (round_number, "bet")
        elif status == RoundStatus.DISCARD:
            pending = engine.get_pending_agari()
            if pending is not None:
                if pending.get("winner_id") == self.client_id:
                    return "agari", (round_number, "agari", len(engine.state.players[0].discards), len(engine.state.players[1].discards))
            elif engine.get_current_player().player_id == self.client_id:
                return "discard", (round_number, "discard", len(me.discards))
        elif status == RoundStatus.ROUND_END_WAITING:
            if self.client_id not in engine.get_next_round_ready_players():
                return "next_round", (round_number, "next_round")
        return None

    def _think_time(self, category: str) -> float:
        think_time = self.profile.config["think_time"]
        low, high = think_time.get(category, think_time["skill"])
        return self._rng.uniform(low, high)

    async def _run(self) -> None:
        while not self._stopped:
            need = self._current_need()
            if need is None:
                return
            category, step = need
            if self._attempts[step] >= self.MAX_ATTEMPTS_PER_STEP:
                return

            await self._sleep(self._think_time(category))

            # 考えている間に状況が変わっていることがあるので調べ直す
            need = self._current_need()
            if need is None or self._stopped:
                return
            category, step = need
            if self._attempts[step] >= self.MAX_ATTEMPTS_PER_STEP:
                logger.warning(
                    "CPU が行動できないため諦めます: match=%s cpu=%s step=%s last_error=%s",
                    self.match_id, self.client_id, step, self._last_error,
                )
                return
            self._attempts[step] += 1
            use_fallback = self._attempts[step] > 1

            try:
                await self._act(category, use_fallback)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("CPU の行動中に例外: match=%s cpu=%s category=%s", self.match_id, self.client_id, category)

    def _observe(self) -> CpuObservation:
        engine = self._session.get_engine(self.match_id)
        return build_observation(engine, self.client_id, self._public)

    async def _send_action(self, action: str, data: dict[str, Any]) -> bool:
        """action を送り、弾かれなかったら True。"""
        logger.info("CPU action: match=%s cpu=%s action=%s data=%s", self.match_id, self.client_id, action, data)
        self._last_error = None
        await self._session.handle_game_action(self.client_id, {"action": action, "data": data})
        if self._last_error is not None:
            logger.info("CPU action rejected: match=%s cpu=%s action=%s error=%s", self.match_id, self.client_id, action, self._last_error)
            return False
        return True

    # ========== 場面ごとの行動 ==========

    async def _act(self, category: str, use_fallback: bool) -> None:
        obs = self._observe()

        if category == "hand_selection":
            await self._act_hand_selection(obs, use_fallback)
        elif category == "mulligan_select":
            await self._select_mulligan(obs, use_fallback)
        elif category == "select_confirm":
            await self._send_action("select_confirm", {"hand_indexes": list(self._pending_confirm or [])})
        elif category == "bet":
            if use_fallback:
                amount = fallback_bet(obs)
            else:
                plan = self._plan or self._brain.evaluate_hand(obs, tuple(obs.me.hand))
                amount = self._brain.choose_bet(obs, plan)
            await self._send_action("bet", {"bet_amount": amount})
        elif category == "discard":
            index = fallback_discard(obs) if use_fallback else self._brain.choose_discard(obs)
            await self._send_action("discard", {"wall_index": index})
        elif category == "agari":
            # エンジンは満貫以上で和了れるときだけロン確認を出すので、必ず受ける
            await self._send_action("agari", {"accept": True})
        elif category == "next_round":
            await self._send_action("next_round", {})

    @staticmethod
    def _pending_from_options(options: dict) -> dict:
        """skill_accepted の mulliganOptions を、engine の pending_mulligan と同じ形にする。"""
        return {"wall_indexes": list(options["targetHandIndexes"]), "options": list(options["options"])}

    async def _select_mulligan(self, obs: CpuObservation, use_fallback: bool) -> None:
        """マリガンの候補から残す 3 枚を選ぶ。候補が分からないとき・打ち直しでは出した牌をそのまま残す。"""
        options = self._mulligan_options
        choices = list(range(MULLIGAN_CANDIDATE_COUNT))
        if options is not None and not use_fallback:
            loop = asyncio.get_running_loop()
            choices = await loop.run_in_executor(
                _SEARCH_EXECUTOR, self._brain.choose_mulligan, obs, self._pending_from_options(options),
            )
        await self._send_action("mulligan_select", {"choices": choices})

    async def _act_hand_selection(self, obs: CpuObservation, use_fallback: bool) -> None:
        if use_fallback:
            await self._send_action("select", {"hand_indexes": fallback_hand_indexes(obs)})
            return

        # 聴牌形の探索は重いので、イベントループを止めないよう別スレッドで行う
        loop = asyncio.get_running_loop()
        plan = await loop.run_in_executor(_SEARCH_EXECUTOR, self._brain.plan_hand, obs)

        for _ in range(self.MAX_SKILL_STEPS):
            data = self._brain.next_skill(obs, plan)
            if data is None:
                break
            skill = data["skill_type"]
            # 弾かれても同じスキルを繰り返さないよう、送る前に数える
            self._public.my_skills_this_round[skill] = self._public.my_skills_this_round.get(skill, 0) + 1

            await self._sleep(self._think_time("skill"))
            need = self._current_need()
            if self._stopped or need is None or need[0] != "hand_selection":
                return

            accepted = await self._send_action("skill", data)
            if accepted and skill == "mulligan":
                await self._select_mulligan(self._observe(), use_fallback=False)
            obs = self._observe()
            if accepted:
                # 交換で山が、透視で見える牌が、役強化で翻数が変わるので、手牌を選び直す
                plan = await loop.run_in_executor(_SEARCH_EXECUTOR, self._brain.replan, obs, plan, skill)

        self._plan = plan
        hand_indexes = list(plan.hand_indexes) if plan is not None else fallback_hand_indexes(obs)
        await self._send_action("select", {"hand_indexes": hand_indexes})
