"""
麻雀牌の2進数表現と文字列変換クラス
牌の情報（5ビット）：
- 萬子 1-9: 0-8 (0b00000 - 0b01000)
- 筒子 1-9: 9-17 (0b01001 - 0b10001)
- 索子 1-9: 18-26 (0b10010 - 0b11010)
- 字牌 東西: 27-28 (0b11011 - 0b11100)

ドラ情報（2ビット）：
    ドラ種別（1ビット）：
        - 0: 通常牌
        - 1: ドラ

    赤ドラ（1ビット）：
        - 0: 通常牌
        - 1: 赤ドラ

合計7ビットで表現
    赤ドラ（1ビット） | ドラ種別（1ビット） | 牌の種類（5ビット）
"""


class TileConverter:
    """麻雀牌の変換クラス"""

    # 牌の種類
    TILE = [
        # 萬子
        "1萬", "2萬", "3萬", "4萬", "5萬", "6萬", "7萬", "8萬", "9萬",
        # 筒子
        "1筒", "2筒", "3筒", "4筒", "5筒", "6筒", "7筒", "8筒", "9筒",
        # 索子
        "1索", "2索", "3索", "4索", "5索", "6索", "7索", "8索", "9索",
        # 字牌
        "東", "西"
    ]

    DORA = ["", "ドラ"]

    AKA = ["", "赤ドラ"]

    @staticmethod
    def binary_to_tile(value: int) -> str:
        """
        2進数値を麻雀牌の文字列に変換

        Args:
            value: 7ビットの整数値（下位5ビット=牌、上位2ビット=ドラ情報）

        Returns:
            麻雀牌の文字列表現

        Examples:
            >>> TileConverter.binary_to_tile(0b0000000)  # 1萬
            '1萬'
            >>> TileConverter.binary_to_tile(0b0100000)  # 1萬(ドラ)
            '1萬(ドラ)'
            >>> TileConverter.binary_to_tile(0b1001101)  # 5筒(赤ドラ)
            '5筒(赤ドラ)'
        """

        # 下位5ビットから牌の種類を取得
        tile_id = value & 0b11111
        # 上位2ビットからドラ情報を取得
        dora = (value >> 5) & 0b1
        aka = (value >> 6) & 0b1

        if tile_id >= len(TileConverter.TILE):
            raise ValueError(f"無効な牌ID: {tile_id}")

        tile_name = TileConverter.TILE[tile_id]
        dora = TileConverter.DORA[dora]
        aka = TileConverter.AKA[aka]

        if dora or aka:
            tile_name = f"{tile_name}({dora}{aka})"

        return tile_name

    @staticmethod
    def array_to_tiles(values: list[int]) -> list[str]:
        """
        数値配列を麻雀牌の文字列配列に変換

        Args:
            values: 7ビット整数のリスト

        Returns:
            麻雀牌の文字列リスト

        Examples:
            >>> TileConverter.array_to_tiles([0, 1, 2])
            ['1萬', '2萬', '3萬']
            >>> TileConverter.array_to_tiles([0b0100000, 0b1001101])
            ['1萬(ドラ)', '5筒(赤ドラ)']
        """

        return [TileConverter.binary_to_tile(v) for v in values]

    @staticmethod
    def tile_to_binary(tile_name: str, dora_type: list[int, int]) -> int:
        """
        麻雀牌の文字列を2進数値に変換

        Args:
            tile_name: 牌の名前（例: "1萬", "東"）
            dora_type: ドラ種別（リスト形式で赤ドラとドラの数を指定）

        Returns:
            7ビットの整数値

        Examples:
            >>> TileConverter.tile_to_binary("1萬")
            0
            >>> TileConverter.tile_to_binary("東", [1, 0])
            0b0111011
            >>> TileConverter.tile_to_binary("5筒", [0, 1])
            0b1001101
        """

        try:
            tile_id = TileConverter.TILE.index(tile_name)
        except ValueError:
            raise ValueError(f"無効な牌名: {tile_name}")

        if not (0 <= dora_type[0] <= 1 and 0 <= dora_type[1] <= 1):
            raise ValueError(f"無効なドラ種別: {dora_type}")

        dora_value = (dora_type[0] << 1) | dora_type[1]
        return tile_id | (dora_value << 5)