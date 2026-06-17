"""
麻雀牌の定義と牌山管理
"""
import random
from typing import List

from .hand_analyzer import HandAnalyzer


class TileDefinitions:
    """牌の定義"""

    # 牌の総数
    TOTAL_TILE_TYPES = 29  # 29種類の牌（萬子1-9、筒子1-9、索子1-9、字牌東西）
    TILES_PER_TYPE = 4
    TOTAL_TILES = TOTAL_TILE_TYPES * TILES_PER_TYPE  # 29種類 × 4枚 = 116枚


class TileWall:
    """牌山を管理するクラス"""

    def __init__(self):
        """牌山を初期化"""
        self.tiles: List[int] = []
        self.dora_id: int = None

        self._initialize_wall()
        self.shuffle()

    def _initialize_wall(self):
        """116枚の牌山を生成"""
        self.tiles = []

        # ドラ表示牌をランダムに選択
        hyouji = random.randrange(0, TileDefinitions.TOTAL_TILE_TYPES)
        if hyouji < 27:
            dora = (hyouji + 1) % 9 + (hyouji // 9) * 9 # ドラは同じ種類の次の牌
        else:
            dora = 55 - hyouji # 字牌のドラは特例で、東→西→東の順でループ
        self.dora_id = dora

        # 各牌を4枚ずつ追加
        for tile_id in range(TileDefinitions.TOTAL_TILE_TYPES):
            for i in range(TileDefinitions.TILES_PER_TYPE):
                if tile_id == hyouji and i == 0:  # ドラ表示牌は1枚だけ抜く
                    continue
                dora_flag = 1 if tile_id == dora else 0 # ドラの場合はフラグを立てる
                dora_flag += 1 << 1 if (tile_id == 4 or tile_id == 13 or tile_id == 22) and i == 0 else 0 # 赤ドラの場合はさらにフラグを立てる

                self.tiles.append(tile_id | (dora_flag << 5)) # ドラ情報を上位2ビットに格納

    def shuffle(self):
        """牌山をシャッフル"""
        random.shuffle(self.tiles)

    def deal(self, count: int = 34) -> tuple[List[int], List[int]]: #, List[List[int]]]:
        """
        牌山から指定枚数を配る

        Args:
            count: 配る枚数 (デフォルト: 34)

        Returns:
            配った牌のリスト
            聴牌形の例
        """

        if count <= 0:
            raise ValueError("count は 1 以上で指定してください。")

        if len(self.tiles) < count:
            raise ValueError(f"牌山の残り枚数が不足しています (need={count}, remain={len(self.tiles)})")

        while True:
            dealt_tiles = self.tiles[:count]
            rest_tiles = self.tiles[count:]

            # 配った 34 枚から満貫以上の聴牌形を探索
            hands = HandAnalyzer.search_tenpai(dealt_tiles, rest_tiles, self.dora_id)
            if hands:
                self.tiles = rest_tiles
                return dealt_tiles, random.choice(hands)

            # 見つからなければ同じ牌山を再シャッフルして再探索
            self.shuffle()

    def reset(self):
        """牌山をリセット"""
        self._initialize_wall()