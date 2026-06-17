"""
手牌の聴牌判定と役計算
"""
from collections import Counter
from functools import lru_cache
from typing import List, Tuple, Generator
from itertools import combinations

from .yaku import Yaku


class HandAnalyzer:
    """手牌の分析・判定を行うクラス"""

    TILE_KIND_COUNT = 29
    TILE_MAX_COUNT = 4

    # ========== 聴牌判定 ==========

    @staticmethod
    def search_tenpai(
        wall: List[int],
        agari_wall: List[int] | None = None,
        dora: int | None = None,
    ) -> List[list[int]]:
        """
        34枚の山牌から聴牌形を検索する

        聴牌形 = 3面子（9枚） + 残り4枚
        - 残り4枚：山牌の残り25枚から選ぶ

        Args:
            wall: 山牌のリスト（34枚を想定）
            agari_wall: 和了判定に使う残り牌のリスト。指定時は満貫以上の聴牌形のみ返す
            dora: ドラの牌ID。agari_wall 指定時に利用する

        Returns:
            聴牌形のリスト
        """
        wall_counter = HandAnalyzer._tiles_to_counter_tuple(wall)
        source_tile_index = HandAnalyzer._build_source_tile_index(wall)
        skip_tiles = tuple(sorted(HandAnalyzer.skip_tenpai_tiles(wall)))
        skip_tile_set = set(skip_tiles)
        results: List[list[int]] = []
        seen: set[Tuple[int, ...]] = set()
        require_mangan = agari_wall is not None and dora is not None
        residual_catalog = HandAnalyzer._tenpai_residual_catalog()

        # 34枚から順番に面子候補を抽出
        mentsu = HandAnalyzer._extract_mentsu_dp(wall_counter, 0, 0)

        for pattern in mentsu:
            removed_wall_counter = HandAnalyzer._subtract_tiles(wall_counter, pattern)
            for rests, waits_all, sparse_counts in residual_catalog:
                if any(removed_wall_counter[tile_id] < need for tile_id, need in sparse_counts):
                    continue

                waiting_tiles = tuple(tile_id for tile_id in waits_all if tile_id not in skip_tile_set)
                if not waiting_tiles:
                    continue

                candidate = tuple(sorted(pattern + rests))
                if candidate in seen:
                    continue

                if require_mangan and not HandAnalyzer._has_mangan_wait(
                    candidate,
                    waiting_tiles,
                    source_tile_index,
                ):
                    continue

                results.append(HandAnalyzer._decorate_hand_from_index(candidate, source_tile_index))
                seen.add(candidate)

        return results

    @staticmethod
    @lru_cache(maxsize=1)
    def _tenpai_residual_catalog() -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[Tuple[int, int], ...]], ...]:
        """4枚残り形のうち聴牌になり得るものを前計算したカタログを返す。"""
        full_counter = tuple([HandAnalyzer.TILE_MAX_COUNT] * HandAnalyzer.TILE_KIND_COUNT)
        residuals = HandAnalyzer._select_tiles_dp(full_counter, 4, 0)
        catalog: list[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[Tuple[int, int], ...]]] = []

        for rests in residuals:
            residual_counter = HandAnalyzer._tiles_to_counter_tuple(rests)
            waits_all = HandAnalyzer._get_waiting_tiles_for_residual(residual_counter, ())
            if not waits_all:
                continue

            sparse_counts = tuple((tile_id, count) for tile_id, count in enumerate(residual_counter) if count > 0)
            catalog.append((rests, waits_all, sparse_counts))

        return tuple(catalog)

    @staticmethod
    def _tiles_to_counter_tuple(tiles: List[int] | Tuple[int, ...]) -> Tuple[int, ...]:
        """牌列を牌種カウントのタプルへ変換する。"""
        counts = [0] * HandAnalyzer.TILE_KIND_COUNT
        for tile in tiles:
            counts[tile & 0b11111] += 1
        return tuple(counts)

    @staticmethod
    def _build_source_tile_index(source_tiles: List[int]) -> Tuple[Tuple[int, ...], ...]:
        """牌種ごとの実牌候補（ドラ/赤ドラ優先）を構築する。"""
        buckets: list[list[int]] = [[] for _ in range(HandAnalyzer.TILE_KIND_COUNT)]
        for tile in source_tiles:
            buckets[tile & 0b11111].append(tile)

        for bucket in buckets:
            bucket.sort(key=lambda t: (((t >> 5) & 0b1) + ((t >> 6) & 0b1)), reverse=True)

        return tuple(tuple(bucket) for bucket in buckets)

    @staticmethod
    def _decorate_hand_from_index(hand: Tuple[int, ...], source_tile_index: Tuple[Tuple[int, ...], ...]) -> List[int]:
        """ベース牌の手牌を、事前計算済み実牌候補から復元する。"""
        selected_counter = Counter(hand)
        decorated: list[int] = []

        for tile_id, count in selected_counter.items():
            decorated.extend(source_tile_index[tile_id][:count])

        return decorated

    @staticmethod
    def _count_hand_bonus_han(hand: Tuple[int, ...], source_tile_index: Tuple[Tuple[int, ...], ...]) -> int:
        """候補手牌に最初から含まれているドラ/赤ドラの翻数だけを返す。"""
        bonus_han = 0

        for tile in HandAnalyzer._decorate_hand_from_index(hand, source_tile_index):
            bonus_han += (tile >> 5) & 0b1
            bonus_han += (tile >> 6) & 0b1

        return bonus_han

    @staticmethod
    def _has_mangan_wait(
        hand: Tuple[int, ...],
        waiting_tiles: Tuple[int, ...],
        source_tile_index: Tuple[Tuple[int, ...], ...],
    ) -> bool:
        """待ち牌のいずれかで満貫以上になるかを返す。"""
        hand_counter = Counter(hand)
        bonus_han = HandAnalyzer._count_hand_bonus_han(hand, source_tile_index)

        for winning_tile in waiting_tiles:
            agari_counter = hand_counter.copy()
            agari_counter[winning_tile] += 1

            if HandAnalyzer._check_mangan_from_counter(agari_counter, winning_tile, bonus_han):
                return True
        return False

    @staticmethod
    def _subtract_tiles(counter_tuple: Tuple[int, ...], tiles: Tuple[int, ...]) -> Tuple[int, ...]:
        """カウントタプルから牌列を減算した新しいタプルを返す。"""
        counts = list(counter_tuple)
        for tile in tiles:
            counts[tile] -= 1
        return tuple(counts)

    @staticmethod
    @lru_cache(maxsize=None)
    def _extract_mentsu_dp(
        counter_tuple: Tuple[int, ...],
        start_tile: int = 0,
        depth: int = 0,
    ) -> Tuple[Tuple[int, ...], ...]:
        """面子3つ分の候補を DP で列挙する。"""
        if depth == 3:
            return ((),)

        results: list[Tuple[int, ...]] = []
        for tile_id in range(start_tile + 1, HandAnalyzer.TILE_KIND_COUNT):
            if counter_tuple[tile_id] <= 0:
                continue

            if counter_tuple[tile_id] >= 3:
                next_counter = list(counter_tuple)
                next_counter[tile_id] -= 3
                for melds in HandAnalyzer._extract_mentsu_dp(tuple(next_counter), tile_id, depth + 1):
                    results.append((tile_id, tile_id, tile_id) + melds)

            elif HandAnalyzer._can_form_run_from_tuple(counter_tuple, tile_id):
                next_counter = list(counter_tuple)
                next_counter[tile_id] -= 1
                next_counter[tile_id + 1] -= 1
                next_counter[tile_id + 2] -= 1
                for melds in HandAnalyzer._extract_mentsu_dp(tuple(next_counter), tile_id, depth + 1):
                    results.append((tile_id, tile_id + 1, tile_id + 2) + melds)

        return tuple(results)

    @staticmethod
    @lru_cache(maxsize=None)
    def _select_tiles_dp(
        counter_tuple: Tuple[int, ...],
        pick_count: int,
        start_tile: int = 0,
    ) -> Tuple[Tuple[int, ...], ...]:
        """カウントタプルから重複なしの牌 multiset を DP で列挙する。"""
        if pick_count == 0:
            return ((),)

        results: list[Tuple[int, ...]] = []
        for tile_id in range(start_tile, HandAnalyzer.TILE_KIND_COUNT):
            max_use = min(counter_tuple[tile_id], pick_count)
            if max_use <= 0:
                continue

            for use_count in range(1, max_use + 1):
                next_counter = list(counter_tuple)
                next_counter[tile_id] -= use_count
                for rests in HandAnalyzer._select_tiles_dp(tuple(next_counter), pick_count - use_count, tile_id):
                    results.append((tile_id,) * use_count + rests)

        return tuple(results)

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_waiting_tiles_from_counter(
        hand_counter: Tuple[int, ...],
        skip_tiles: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        """13枚手牌カウントから待ち牌を返す。"""
        skip_tile_set = set(skip_tiles)
        waiting_tiles: list[int] = []
        for tile_id in range(HandAnalyzer.TILE_KIND_COUNT):
            if tile_id in skip_tile_set:
                continue

            augmented = list(hand_counter)
            augmented[tile_id] += 1
            if HandAnalyzer._is_win_tuple(tuple(augmented)):
                waiting_tiles.append(tile_id)

        return tuple(waiting_tiles)

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_waiting_tiles_for_residual(
        residual_counter: Tuple[int, ...],
        skip_tiles: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        """3面子を除いた4枚残りから待ち牌を返す。"""
        skip_tile_set = set(skip_tiles)
        waiting_tiles: list[int] = []
        for tile_id in range(HandAnalyzer.TILE_KIND_COUNT):
            if tile_id in skip_tile_set:
                continue

            augmented = list(residual_counter)
            augmented[tile_id] += 1
            if HandAnalyzer._is_meld_plus_head(tuple(augmented)):
                waiting_tiles.append(tile_id)

        return tuple(waiting_tiles)

    @staticmethod
    @lru_cache(maxsize=None)
    def _is_win_tuple(counter_tuple: Tuple[int, ...]) -> bool:
        """牌種カウントタプルから和了形かどうかを判定する。"""
        if HandAnalyzer._is_titoitsu_tuple(counter_tuple):
            return True

        for tile_id, count in enumerate(counter_tuple):
            if count < 2:
                continue

            next_counter = list(counter_tuple)
            next_counter[tile_id] -= 2
            if HandAnalyzer._can_form_all_melds(tuple(next_counter)):
                return True

        return False

    @staticmethod
    @lru_cache(maxsize=None)
    def _is_meld_plus_head(counter_tuple: Tuple[int, ...]) -> bool:
        """5枚が 1 面子 + 1 雀頭へ分解できるかを判定する。"""
        for tile_id, count in enumerate(counter_tuple):
            if count < 2:
                continue

            next_counter = list(counter_tuple)
            next_counter[tile_id] -= 2
            if HandAnalyzer._is_single_meld(tuple(next_counter)):
                return True

        return False

    @staticmethod
    @lru_cache(maxsize=None)
    def _is_single_meld(counter_tuple: Tuple[int, ...]) -> bool:
        """3枚がちょうど1面子かどうかを判定する。"""
        tile_id = None
        total = 0
        for tid, count in enumerate(counter_tuple):
            total += count
            if tile_id is None and count > 0:
                tile_id = tid

        if total != 3 or tile_id is None:
            return False

        if counter_tuple[tile_id] == 3:
            return True

        return (
            HandAnalyzer._can_form_run_from_tuple(counter_tuple, tile_id)
            and counter_tuple[tile_id] == 1
            and counter_tuple[tile_id + 1] == 1
            and counter_tuple[tile_id + 2] == 1
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def _can_form_all_melds(counter_tuple: Tuple[int, ...]) -> bool:
        """残り牌がすべて面子へ分解できるかを DP で判定する。"""
        tile_id = None
        for tid, count in enumerate(counter_tuple):
            if count > 0:
                tile_id = tid
                break

        if tile_id is None:
            return True

        if counter_tuple[tile_id] >= 3:
            next_counter = list(counter_tuple)
            next_counter[tile_id] -= 3
            if HandAnalyzer._can_form_all_melds(tuple(next_counter)):
                return True

        if HandAnalyzer._can_form_run_from_tuple(counter_tuple, tile_id):
            next_counter = list(counter_tuple)
            next_counter[tile_id] -= 1
            next_counter[tile_id + 1] -= 1
            next_counter[tile_id + 2] -= 1
            if HandAnalyzer._can_form_all_melds(tuple(next_counter)):
                return True

        return False

    @staticmethod
    def _can_form_run_from_tuple(counter_tuple: Tuple[int, ...], tile_id: int) -> bool:
        """カウントタプル上で順子を作れるかどうかの判定。"""
        if not HandAnalyzer._is_suited(tile_id):
            return False
        if tile_id % 9 >= 7:
            return False
        return counter_tuple[tile_id + 1] > 0 and counter_tuple[tile_id + 2] > 0

    @staticmethod
    def _is_titoitsu_tuple(counter_tuple: Tuple[int, ...]) -> bool:
        """牌種カウントタプルから七対子を判定する。"""
        return all(count == 2 for count in counter_tuple if count > 0)

    @staticmethod
    def _extract_mentsu(wall_counter: Counter, start_tile: int = 0, depth: int = 0) -> Generator[list[int], None, None]:
        """
        山牌から面子候補を抽出する

        Args:
            wall_counter: 山牌のカウンター
            start_tile: 探索を開始する牌ID
            depth: 再帰の深さ

        Returns:
            面子候補のリスト
        """
        if depth == 3:  # 面子が3つできたら終了
            yield []
            return

        for tile_id in sorted(wall_counter.keys()):
            if start_tile >= tile_id:
                continue

            temp_counter = wall_counter.copy()

            # 刻子
            if temp_counter[tile_id] >= 3:
                temp_counter[tile_id] -= 3
                for melds in HandAnalyzer._extract_mentsu(temp_counter, tile_id, depth + 1):
                    yield [tile_id] * 3 + melds

            # 順子
            elif HandAnalyzer._can_form_run(temp_counter, tile_id):
                temp_counter[tile_id] -= 1
                temp_counter[tile_id + 1] -= 1
                temp_counter[tile_id + 2] -= 1
                for melds in HandAnalyzer._extract_mentsu(temp_counter, tile_id, depth + 1):
                    yield [tile_id, tile_id + 1, tile_id + 2] + melds

    @staticmethod
    def is_tenpai(hand: List[int], wall: List[int]) -> bool:
        """
        聴牌判定

        Args:
            hand: 手牌のリスト（13枚を想定）
            wall: 山牌のリスト

        Returns:
            聴牌かどうか
        """
        skip_tiles = tuple(sorted(HandAnalyzer.skip_tenpai_tiles(wall)))
        hand_counter = HandAnalyzer._tiles_to_counter_tuple(hand)
        return bool(HandAnalyzer._get_waiting_tiles_from_counter(hand_counter, skip_tiles))

    @staticmethod
    def get_tenpai_waiting_tiles(hand: List[int], wall: List[int]) -> List[int]:
        """
        待ち牌の検索

        Args:
            hand: 手牌のリスト（13枚を想定）
            wall: 山牌のリスト

        Returns:
            待ち牌のリスト
        """
        skip_tiles = tuple(sorted(HandAnalyzer.skip_tenpai_tiles(wall)))
        hand_counter = HandAnalyzer._tiles_to_counter_tuple(hand)
        return list(HandAnalyzer._get_waiting_tiles_from_counter(hand_counter, skip_tiles))

    @staticmethod
    def without_hand(hand: List[int], wall: List[int]) -> List[int]:
        """
        手牌を除外した山牌のリストを返す

        Args:
            hand: 手牌のリスト
            wall: 山牌のリスト

        Returns:
            手牌を除外した山牌のリスト
        """
        hand_counter = Counter(hand)
        wall_counter = Counter(wall)

        for tile_id, count in hand_counter.items():
            wall_counter[tile_id] -= count

        return list(wall_counter.elements())

    @staticmethod
    def skip_tenpai_tiles(wall: List[int]) -> List[int]:
        """
        聴牌判定で待ち牌から除外する牌IDのリストを返す

        Args:
            wall: 山牌のリスト

        Returns:
            待ち牌から除外する牌IDのリスト
        """
        wall_counter = Counter(t & 0b11111 for t in wall)
        return [tile_id for tile_id, count in wall_counter.items() if count >= 4]

    @staticmethod
    def is_win(hand: List[int]) -> bool:
        """
        和了形かどうかの判定

        Args:
            hand: 手牌のリスト（14枚を想定）

        Returns:
            和了形かどうか
        """
        counter = Counter(t & 0b11111 for t in hand)
        return HandAnalyzer._is_win(counter)

    @staticmethod
    def _is_win(counter: Counter) -> bool:
        """和了形かどうかの判定"""
        if HandAnalyzer._is_titoitsu(counter):
            return True

        for tile_id in HandAnalyzer._head_candidates(counter):
            temp_counter = counter.copy()
            temp_counter[tile_id] -= 2
            if any(len(melds) == 4 for melds in HandAnalyzer._generate_melds(temp_counter)):
                return True

        return False

    # ========== 役計算 ==========

    
    @staticmethod
    def filter_mangan_hands(hands: list[list[int]], wall: List[int], dora: int) -> List[list[int]]:
        """
        聴牌形の手牌リストのうち
        満貫以上となる上がりを持つものを返す

        Args:
            hands: 聴牌形の手牌リスト（13枚を想定）
            wall: 山牌のリスト
            dora: ドラの牌ID
        Returns:
            満貫以上の聴牌形の手牌リスト
        """

        mangan_hands = []
        for hand in hands:
            waiting_tiles = tuple(HandAnalyzer.get_tenpai_waiting_tiles(hand, wall))
            if HandAnalyzer._has_mangan_wait(tuple(t & 0b11111 for t in hand), waiting_tiles, wall, dora):
                mangan_hands.append(hand)

        return mangan_hands

    @staticmethod
    def check_mangan(hand: List[int], winning_tile: int | None = None) -> bool:
        """
        満貫以上かどうかの判定

        Args:
            hand: 手牌（14枚を想定）
            winning_tile: 上がり牌の牌ID。ロン時の暗刻判定補正に利用する

        Returns:
            満貫（４翻）以上かどうか
        """
        base_tiles = [t & 0b11111 for t in hand]
        counter = Counter(base_tiles)
        bonus_han = sum(1 for t in hand if ((t >> 5) & 0b1 == 1) or ((t >> 6) & 0b1 == 1))
        resolved_winning_tile = (winning_tile & 0b11111) if winning_tile is not None else None
        return HandAnalyzer._check_mangan_from_counter(counter, resolved_winning_tile, bonus_han)

    @staticmethod
    def _check_mangan_from_counter(
        counter: Counter,
        winning_tile: int | None,
        bonus_han: int,
    ) -> bool:
        """基底カウントとボーナス翻数から満貫以上かを判定する。"""
        counter_tuple = tuple(counter.get(i, 0) for i in range(HandAnalyzer.TILE_KIND_COUNT))
        return HandAnalyzer._check_mangan_from_counter_tuple(counter_tuple, winning_tile, bonus_han)

    @staticmethod
    @lru_cache(maxsize=None)
    def _check_mangan_from_counter_tuple(
        counter_tuple: Tuple[int, ...],
        winning_tile: int | None,
        bonus_han: int,
    ) -> bool:
        """基底カウントとボーナス翻数から満貫以上かを判定する（キャッシュ版）。"""
        target = 4 - bonus_han

        if target <= 0:
            return True

        return HandAnalyzer._max_han_from_counter_tuple(counter_tuple, winning_tile) >= target

    @staticmethod
    @lru_cache(maxsize=None)
    def _max_han_from_counter_tuple(
        counter_tuple: Tuple[int, ...],
        winning_tile: int | None,
    ) -> int:
        """ボーナス抜きの最大翻数を返す（キャッシュ版）。"""
        counter = Counter({i: c for i, c in enumerate(counter_tuple) if c > 0})
        best_han = 0

        # 七対子判定
        if HandAnalyzer._is_titoitsu(counter):
            han = 3  # 七対子 + 立直
            if HandAnalyzer._is_tanyao(counter):
                han += 1
            if HandAnalyzer._is_chinitsu(counter):
                han += 6
            elif HandAnalyzer._is_honitsu(counter):
                han += 3
            if HandAnalyzer._is_honroutou(counter):
                han += 2
            if han > best_han:
                best_han = han

        # 通常手（4面子1雀頭）の全分解を探索
        for head_tile in HandAnalyzer._head_candidates(counter):
            temp_counter = counter.copy()
            temp_counter[head_tile] -= 2

            for melds in HandAnalyzer._generate_melds(temp_counter):
                if len(melds) != 4:
                    continue
                han = HandAnalyzer._evaluate_melds_han(counter, melds, head_tile, winning_tile)
                if han > best_han:
                    best_han = han

        return best_han

    @staticmethod
    def _evaluate_melds_han(
        counter: Counter,
        melds: List[Tuple[str, int]],
        head: int,
        winning_tile: int | None = None,
    ) -> int:
        """面子と雀頭から役の合計翻数だけを返す。"""
        # 役満
        if HandAnalyzer._is_junsei_churen_poutou(counter, winning_tile):
            return 26
        if HandAnalyzer._is_churen_poutou(counter):
            return 13
        if HandAnalyzer._is_ryuuisou(counter):
            return 13
        if HandAnalyzer._is_chinroutou(counter):
            return 13
        if HandAnalyzer._is_suuankou(melds, winning_tile):
            return 13

        han = 1  # 立直

        if HandAnalyzer._is_tanyao(counter):
            han += 1
        if HandAnalyzer._is_pinfu(melds, head):
            han += 1
        if HandAnalyzer._is_chinitsu(counter):
            han += 6
        elif HandAnalyzer._is_honitsu(counter):
            han += 3
        if HandAnalyzer._is_honroutou(counter):
            han += 2

        if HandAnalyzer._is_junchan(melds, head):
            han += 3
        elif HandAnalyzer._is_chanta(melds, head):
            han += 2

        # NOTE:
        # `一気通貫` は現在の Yaku テーブルに未登録のため、
        # 高速判定側で加点すると本採点と乖離して偽陽性になる。
        if HandAnalyzer._is_sanshoku_doujun(melds):
            han += 2
        if HandAnalyzer._is_sanshoku_doukou(melds):
            han += 2
        if HandAnalyzer._is_toitoi(melds):
            han += 2
        if HandAnalyzer._is_sanankou(melds, winning_tile):
            han += 2

        if HandAnalyzer._is_ryanpeikou(melds):
            han += 3
        elif HandAnalyzer._is_ipeikou(melds):
            han += 1

        if HandAnalyzer._is_ton(melds):
            han += 1
        if HandAnalyzer._is_sha(melds):
            han += 1

        return han

    @staticmethod
    def calc_yaku(hand: List[int], winning_tile: int | None = None) -> int:
        """
        役の計算

        Args:
            hand: 手牌（14枚を想定）
            winning_tile: 上がり牌の牌ID。ロン時の暗刻判定補正に利用する

        Returns:
            役の合計翻数
        """
        yaku = HandAnalyzer.enum_yaku(hand, winning_tile=winning_tile)
        return sum(Yaku.get_han_by_name(name) for name in yaku)


    @staticmethod
    def enum_yaku(hand: List[int], winning_tile: int | None = None) -> List[str]:
        """
        役の列挙

        Args:
            hand: 手牌（14枚を想定）
            winning_tile: 上がり牌の牌ID。ロン時の暗刻判定補正に利用する

        Returns:
            役名のリスト
        """
        base_tiles = [t & 0b11111 for t in hand]
        counter = Counter(base_tiles)

        # ドラと赤ドラの枚数
        dora_count = sum(1 for t in hand if (t >> 5) & 0b1 == 1)
        aka_count = sum(1 for t in hand if (t >> 6) & 0b1 == 1)
        dora_yaku = ["ドラ"] * dora_count + ["赤ドラ"] * aka_count
        dora_han = dora_count + aka_count

        best_yaku: List[str] = []
        best_han = 0

        # 七対子判定
        if HandAnalyzer._is_titoitsu(counter):
            yaku = ["七対子"]
            han = 2
            if HandAnalyzer._is_tanyao(counter):
                yaku.append("断么九")
                han += 1
            if HandAnalyzer._is_chinitsu(counter):
                yaku.append("清一色")
                han += 6
            elif HandAnalyzer._is_honitsu(counter):
                yaku.append("混一色")
                han += 3
            if HandAnalyzer._is_honroutou(counter):
                yaku.append("混老頭")
                han += 2

            yaku.append("立直")
            han += 1

            best_yaku = yaku + dora_yaku
            best_han = han + dora_han

        # 通常手（4面子1雀頭）の全分解を探索
        for head_tile in HandAnalyzer._head_candidates(counter):
            temp_counter = counter.copy()
            temp_counter[head_tile] -= 2

            for melds in HandAnalyzer._generate_melds(temp_counter):
                if len(melds) != 4:
                    continue

                yaku, han = HandAnalyzer._evaluate_melds(counter, melds, head_tile, winning_tile)
                yaku += dora_yaku
                han += dora_han

                if han > best_han:
                    best_han = han
                    best_yaku = yaku

        return best_yaku

    @staticmethod
    def _evaluate_melds(
        counter: Counter,
        melds: List[Tuple[str, int]],
        head: int,
        winning_tile: int | None = None,
    ) -> Tuple[List[str], int]:
        """面子と雀頭から役を判定する"""
        yaku: List[str] = []
        han = 0

        # 役満
        if HandAnalyzer._is_junsei_churen_poutou(counter, winning_tile):
            return ["純正九蓮宝燈"], 26
        elif HandAnalyzer._is_churen_poutou(counter):
            return ["九蓮宝燈"], 13
        elif HandAnalyzer._is_ryuuisou(counter):
            return ["緑一色"], 13
        elif HandAnalyzer._is_chinroutou(counter):
            return ["清老頭"], 13
        elif HandAnalyzer._is_suuankou(melds, winning_tile):
            return ["四暗刻"], 13

        if HandAnalyzer._is_tanyao(counter):
            yaku.append("断么九")
            han += 1
        if HandAnalyzer._is_pinfu(melds, head):
            yaku.append("平和")
            han += 1
        if HandAnalyzer._is_chinitsu(counter):
            yaku.append("清一色")
            han += 6
        elif HandAnalyzer._is_honitsu(counter):
            yaku.append("混一色")
            han += 3
        if HandAnalyzer._is_honroutou(counter):
            yaku.append("混老頭")
            han += 2

        if HandAnalyzer._is_junchan(melds, head):
            yaku.append("純全帯么九")
            han += 3
        elif HandAnalyzer._is_chanta(melds, head):
            yaku.append("混全帯么九")
            han += 2

        if HandAnalyzer._is_ikkitsuukan(melds):
            yaku.append("一気通貫")
            han += 2
        if HandAnalyzer._is_sanshoku_doujun(melds):
            yaku.append("三色同順")
            han += 2
        if HandAnalyzer._is_sanshoku_doukou(melds):
            yaku.append("三色同刻")
            han += 2
        if HandAnalyzer._is_toitoi(melds):
            yaku.append("対々和")
            han += 2
        if HandAnalyzer._is_sanankou(melds, winning_tile):
            yaku.append("三暗刻")
            han += 2

        if HandAnalyzer._is_ryanpeikou(melds):
            yaku.append("二盃口")
            han += 3
        elif HandAnalyzer._is_ipeikou(melds):
            yaku.append("一盃口")
            han += 1

        if HandAnalyzer._is_ton(melds):
            yaku.append("東")
            han += 1
        if HandAnalyzer._is_sha(melds):
            yaku.append("西")
            han += 1

        yaku.append("立直")
        han += 1

        return yaku, han

    # ========== 面子分解 ==========

    @staticmethod
    def _head_candidates(counter: Counter) -> List[int]:
        """対子候補の牌IDを返す"""
        return [tile_id for tile_id, count in counter.items() if count >= 2]

    @staticmethod
    def _generate_melds(counter: Counter) -> List[List[Tuple[str, int]]]:
        """面子分解を全探索して返す"""
        tile_id = None
        for tid in range(29):
            if counter[tid] > 0:
                tile_id = tid
                break

        if tile_id is None:
            return [[]]

        results: List[List[Tuple[str, int]]] = []

        # 刻子
        if counter[tile_id] >= 3:
            counter[tile_id] -= 3
            for rest in HandAnalyzer._generate_melds(counter):
                results.append([("triplet", tile_id)] + rest)
            counter[tile_id] += 3

        # 順子
        if HandAnalyzer._can_form_run(counter, tile_id):
            counter[tile_id] -= 1
            counter[tile_id + 1] -= 1
            counter[tile_id + 2] -= 1
            for rest in HandAnalyzer._generate_melds(counter):
                results.append([("run", tile_id)] + rest)
            counter[tile_id] += 1
            counter[tile_id + 1] += 1
            counter[tile_id + 2] += 1

        return results

    @staticmethod
    def _can_form_run(counter: Counter, tile_id: int) -> bool:
        """順子を作れるかどうかの判定"""
        if not HandAnalyzer._is_suited(tile_id):
            return False
        if tile_id % 9 >= 7:
            return False
        return counter[tile_id + 1] > 0 and counter[tile_id + 2] > 0

    # ========== 基本判定（牌単位） ==========

    @staticmethod
    def _is_suited(tile_id: int) -> bool:
        """数牌かどうかの判定"""
        return tile_id < 27

    @staticmethod
    def _is_honor(tile_id: int) -> bool:
        """字牌かどうかの判定"""
        return tile_id >= 27

    @staticmethod
    def _is_terminal(tile_id: int) -> bool:
        """么九牌かどうかの判定"""
        return tile_id % 9 == 0 or tile_id % 9 == 8

    @staticmethod
    def _tile_suit(tile_id: int) -> int:
        """牌の種類を返す（萬子=0、筒子=1、索子=2、字牌=3）"""
        return min(tile_id // 9, 3)

    # ========== 役判定（手全体） ==========

    @staticmethod
    def _is_titoitsu(counter: Counter) -> bool:
        """七対子の判定"""
        return all(count == 2 for count in counter.values())

    @staticmethod
    def _is_tanyao(counter: Counter) -> bool:
        """断么九かどうかの判定"""
        return all(not HandAnalyzer._is_terminal(tile_id) for tile_id in counter.keys())

    @staticmethod
    def _is_pinfu(melds: List[Tuple[str, int]], head: int) -> bool:
        """平和かどうかの判定"""
        if HandAnalyzer._is_honor(head):
            return False
        return all(kind == "run" for kind, _ in melds)

    @staticmethod
    def _is_honitsu(counter: Counter) -> bool:
        """混一色かどうかの判定"""
        suits = {HandAnalyzer._tile_suit(tile_id) for tile_id in counter.keys() if not HandAnalyzer._is_honor(tile_id)}
        return len(suits) == 1 and any(HandAnalyzer._is_honor(tile_id) for tile_id in counter.keys())

    @staticmethod
    def _is_chinitsu(counter: Counter) -> bool:
        """清一色かどうかの判定"""
        suits = {HandAnalyzer._tile_suit(tile_id) for tile_id in counter.keys() if not HandAnalyzer._is_honor(tile_id)}
        return len(suits) == 1 and all(not HandAnalyzer._is_honor(tile_id) for tile_id in counter.keys())

    @staticmethod
    def _is_honroutou(counter: Counter) -> bool:
        """混老頭かどうかの判定"""
        return all(HandAnalyzer._is_terminal(tile_id) or HandAnalyzer._is_honor(tile_id) for tile_id in counter.keys())

    @staticmethod
    def _is_chinroutou(counter: Counter) -> bool:
        """清老頭かどうかの判定"""
        all_tile = all(HandAnalyzer._is_terminal(tile_id) for tile_id in counter.keys())
        all_honor = all(not HandAnalyzer._is_honor(tile_id) for tile_id in counter.keys())
        return all_tile and all_honor


    @staticmethod
    def _is_chanta(melds: List[Tuple[str, int]], head: int) -> bool:
        """混全帯么九かどうかの判定"""
        if not HandAnalyzer._is_honor(head) and not HandAnalyzer._is_terminal(head):
            return False
        for kind, tile_id in melds:
            if kind == "run" and (tile_id % 9 != 0 and tile_id % 9 != 7):
                return False
            elif kind == "triplet" and not (HandAnalyzer._is_honor(tile_id) or HandAnalyzer._is_terminal(tile_id)):
                return False
        return True

    @staticmethod
    def _is_junchan(melds: List[Tuple[str, int]], head: int) -> bool:
        """純全帯么九かどうかの判定"""
        if not HandAnalyzer._is_terminal(head):
            return False
        for kind, tile_id in melds:
            if kind == "run" and (tile_id % 9 != 0 and tile_id % 9 != 7):
                return False
            elif kind == "triplet" and not HandAnalyzer._is_terminal(tile_id):
                return False
        return True

    @staticmethod
    def _is_ryuuisou(counter: Counter) -> bool:
        """緑一色かどうかの判定"""
        green_tiles = {19, 20, 21, 23, 25}
        return all(tile_id in green_tiles for tile_id in counter.keys())

    @staticmethod
    def _is_churen_poutou(counter: Counter) -> bool:
        """九蓮宝燈かどうかの判定"""
        for suit_base in (0, 9, 18):
            if all(counter[suit_base + i] >= (3 if i in (0, 8) else 1) for i in range(9)):
                return True
        return False

    @staticmethod
    def _is_junsei_churen_poutou(counter: Counter, winning_tile: int | None = None) -> bool:
        """純正九蓮宝燈かどうかの判定"""
        if winning_tile is None:
            return False

        winning_tile_id = winning_tile & 0b11111
        suit_base = (winning_tile_id // 9) * 9

        if suit_base >= 27:
            return False

        if any(counter[tile_id] > 0 for tile_id in range(27, 29)):
            return False

        if any(counter[tile_id] > 0 for tile_id in range(27) if tile_id < suit_base or tile_id >= suit_base + 9):
            return False

        base_pattern = [3, 1, 1, 1, 1, 1, 1, 1, 3]
        suit_counts = [counter[suit_base + i] for i in range(9)]
        suit_counts[winning_tile_id - suit_base] -= 1

        return suit_counts == base_pattern

    # ========== 役判定（面子単位） ==========

    @staticmethod
    def _is_suuankou(melds: List[Tuple[str, int]], winning_tile: int | None = None) -> bool:
        """四暗刻かどうかの判定"""
        return HandAnalyzer._count_concealed_triplets(melds, winning_tile) == 4

    @staticmethod
    def _is_sanankou(melds: List[Tuple[str, int]], winning_tile: int | None = None) -> bool:
        """三暗刻かどうかの判定"""
        return HandAnalyzer._count_concealed_triplets(melds, winning_tile) == 3

    @staticmethod
    def _is_toitoi(melds: List[Tuple[str, int]]) -> bool:
        """対々和かどうかの判定"""
        return all(kind == "triplet" for kind, _ in melds)

    @staticmethod
    def _count_concealed_triplets(melds: List[Tuple[str, int]], winning_tile: int | None = None) -> int:
        """上がり牌で完成した刻子を除いた暗刻数を返す"""
        concealed_triplets = sum(1 for kind, _ in melds if kind == "triplet")
        if winning_tile is None:
            return concealed_triplets

        winning_tile_id = winning_tile & 0b11111
        # シャンポン待ちのロンでは、和了牌で完成した刻子だけを暗刻から外す。
        if any(kind == "triplet" and tile_id == winning_tile_id for kind, tile_id in melds):
            concealed_triplets -= 1

        return concealed_triplets

    @staticmethod
    def _is_ikkitsuukan(melds: List[Tuple[str, int]]) -> bool:
        """一気通貫かどうかの判定"""
        run_starts = {start for kind, start in melds if kind == "run"}
        for suit_base in (0, 9, 18):
            if {suit_base, suit_base + 3, suit_base + 6}.issubset(run_starts):
                return True
        return False

    @staticmethod
    def _is_sanshoku_doujun(melds: List[Tuple[str, int]]) -> bool:
        """三色同順かどうかの判定"""
        run_starts = {start for kind, start in melds if kind == "run"}
        for base in range(7):
            if {base, base + 9, base + 18}.issubset(run_starts):
                return True
        return False

    @staticmethod
    def _is_sanshoku_doukou(melds: List[Tuple[str, int]]) -> bool:
        """三色同刻かどうかの判定"""
        triplet_tiles = {tile_id for kind, tile_id in melds if kind == "triplet"}
        for base in range(9):
            if {base, base + 9, base + 18}.issubset(triplet_tiles):
                return True
        return False

    @staticmethod
    def _is_ipeikou(melds: List[Tuple[str, int]]) -> bool:
        """一盃口かどうかの判定"""
        run_counts = Counter(start for kind, start in melds if kind == "run")
        return any(count >= 2 for count in run_counts.values())

    @staticmethod
    def _is_ryanpeikou(melds: List[Tuple[str, int]]) -> bool:
        """二盃口かどうかの判定"""
        run_counts = Counter(start for kind, start in melds if kind == "run")
        return sum(count >= 2 for count in run_counts.values()) >= 2

    @staticmethod
    def _is_ton(melds: List[Tuple[str, int]]) -> bool:
        """東の刻子かどうかの判定"""
        return any(kind == "triplet" and tile_id == 27 for kind, tile_id in melds)

    @staticmethod
    def _is_sha(melds: List[Tuple[str, int]]) -> bool:
        """西の刻子かどうかの判定"""
        return any(kind == "triplet" and tile_id == 28 for kind, tile_id in melds)