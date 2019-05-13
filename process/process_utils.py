import re


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def is_cjk_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((0x3040 <= cp <= 0x30FF) or  # 日文
            (0x4E00 <= cp <= 0x9FFF) or  #
            (0x3400 <= cp <= 0x4DBF) or  #
            (0xAC00 <= cp <= 0xD7AF) or  # 韩文
            (0x20000 <= cp <= 0x2A6DF) or  #
            (0x2A700 <= cp <= 0x2B73F) or  #
            (0x2B740 <= cp <= 0x2B81F) or  #
            (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or  #
            (0x2F800 <= cp <= 0x2FA1F)):  #
        return True
    return False


def is_chinese_char(cp):
    if ((0x4E00 <= cp <= 0x9FFF) or  #
            (0x3400 <= cp <= 0x4DBF) or  #
            (0x20000 <= cp <= 0x2A6DF) or  #
            (0x2A700 <= cp <= 0x2B73F) or  #
            (0x2B740 <= cp <= 0x2B81F) or  #
            (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or  #
            (0x2F800 <= cp <= 0x2FA1F)):  #
        return True
    return False


def is_punctuation(cp):
    if (
            (0x0020 <= cp <= 0x002F) or
            (0x003A <= cp <= 0x0040) or
            (0x005B <= cp <= 0x0060) or
            (0x007B <= cp <= 0x007E) or
            (0x00A0 <= cp <= 0x00BF) or
            (0x2000 <= cp <= 0x206F) or  #
            (0x2E00 <= cp <= 0x2E7F) or  #
            (0x3000 <= cp <= 0x303F) or
            (0xFE30 <= cp <= 0xFE4F) or
            (0xFE50 <= cp <= 0xFE6F)
    ):  #
        return True
    return False


rENUM = re.compile(r'(([-–+])?\d+(([.·])\d+)?%?|([0-9_.·]*[A-Za-z]+[0-9_.·]*)+)')
rNUM = re.compile(r'([-–+])?\d+(([.·])\d+)?%?')
rENG = re.compile(r'([0-9_.·]*[A-Za-z]+[0-9_.·]*)+')