from enum import Enum


class Yaku(Enum):
    """麻雀役の列挙型（役名、翻数）"""

    # ========== 基本役（1翻） ==========
    RICHI = ("立直", 1)
    TANYAO = ("断么九", 1)
    HEIKO = ("平和", 1)
    IPEIKOU = ("一盃口", 1)
    YAKUNIN_EAST = ("東", 1)
    YAKUNIN_WEST = ("西", 1)
    DORA = ("ドラ", 1)
    AKA_DORA = ("赤ドラ", 1)
    IPPATSU = ("一発", 1)
    KAWA_ZO = ("河底撈魚", 1)

    # ========== 2翻役 ==========
    SANSHOKU_DOUJUN = ("三色同順", 2)
    SANSHOKU_DOUKOU = ("三色同刻", 2)
    SANANKOU = ("三暗刻", 2)
    TOITOI = ("対々和", 2)
    HONROUTOU = ("混老頭", 2)
    CHANTA = ("混全帯么九", 2)
    CHIITOI = ("七対子", 2)
    IKKITSUUKAN = ("一気通貫", 2)

    # ========== 3翻役 ==========
    RYANPEIKOU = ("二盃口", 3)
    HONITSU = ("混一色", 3)
    JUNCHAN = ("純全帯么九", 3)

    # ========== 6翻役 ==========
    CHINITSU = ("清一色", 6)

    # ========== 役満（13翻） ==========
    CHUREN_POUTOU = ("九蓮宝燈", 13)
    RYUUISOU = ("緑一色", 13)
    CHINROUTOU = ("清老頭", 13)
    SUUANKOU = ("四暗刻", 13)

    #========= 二倍役満（26翻） ==========
    JUNSEI_CHUREN_POUTOU = ("純正九蓮宝燈", 26)

    @property
    def japanese_name(self) -> str:
        """役の日本語名"""
        return self.value[0]

    @property
    def han(self) -> int:
        """役の翻数"""
        return self.value[1]

    def __str__(self) -> str:
        """文字列表現は日本語名"""
        return self.japanese_name

    @classmethod
    def get_han_by_name(cls, name: str) -> int:
        """
        役名から翻数を取得

        Args:
            name: 役の日本語名

        Returns:
            役の翻数。見つからない場合は -1

        Examples:
            >>> Yaku.get_han_by_name("立直")
            1
            >>> Yaku.get_han_by_name("清一色")
            6
            >>> Yaku.get_han_by_name("存在しない役")
            -1
        """
        if name == "河底":
            name = cls.KAWA_ZO.japanese_name

        for member in cls:
            if member.japanese_name == name:
                return member.han
        return -1

    @classmethod
    def get_all_names_by_han(cls, han: int) -> list[str]:
        """
        翻数から該当する役名のリストを取得

        Args:
            han: 翻数

        Returns:
            指定翻数の役名リスト

        Examples:
            >>> Yaku.get_all_names_by_han(1)
            ['立直', '断么九', '平和', ...]
            >>> Yaku.get_all_names_by_han(13)
            ['九蓮宝燈', '緑一色', '清老頭', '四暗刻']
        """
        return [member.japanese_name for member in cls if member.han == han]
