# preprocess.py
import re
import unicodedata
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------- PII 掩码 ----------
class PIIMask:
    """PII 模式及掩码规则"""
    PATTERNS = [
        (r'\b\d{17}[\dXx]\b', '[ID_NUMBER]'),          # 身份证
        (r'\b1[3-9]\d{9}\b', '[PHONE]'),               # 手机号
        (r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]'),    # 邮箱
        (r'\b\d{16,19}\b', '[BANK_CARD]'),             # 银行卡
        (r'\b\d{4}-\d{2}-\d{2}\b', '[DATE]'),          # 日期可选掩码
    ]

    @classmethod
    def mask(cls, text: str) -> str:
        for pattern, repl in cls.PATTERNS:
            text = re.sub(pattern, repl, text)
        return text


# ---------- 文本规范化 ----------
class TextNormalizer:
    """文本规范化工具：Unicode标准化、全角转半角、零宽字符移除、空白归一化等"""

    @staticmethod
    def unicode_normalize(text: str, form: str = 'NFKC') -> str:
        """Unicode 标准化 (NFKC 兼容分解并组合)"""
        return unicodedata.normalize(form, text)

    @staticmethod
    def fullwidth_to_halfwidth(text: str) -> str:
        """全角字符转半角（字母、数字、空格、标点）"""
        def _convert(ch: str) -> str:
            code = ord(ch)
            if 0xFF01 <= code <= 0xFF5E:          # 全角 ! 到 ~
                return chr(code - 0xFEE0)
            elif code == 0x3000:                  # 全角空格
                return ' '
            else:
                return ch
        return ''.join(_convert(ch) for ch in text)

    @staticmethod
    def remove_zero_width_chars(text: str) -> str:
        """移除零宽字符（如零宽连接符、零宽空格等）"""
        zero_width_chars = [
            '\u200b',  # ZERO WIDTH SPACE
            '\u200c',  # ZERO WIDTH NON-JOINER
            '\u200d',  # ZERO WIDTH JOINER
            '\ufeff',  # ZERO WIDTH NO-BREAK SPACE (BOM)
            '\u200e', '\u200f'  # LEFT-TO-RIGHT MARK, RIGHT-TO-LEFT MARK
        ]
        for zc in zero_width_chars:
            text = text.replace(zc, '')
        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """将各种空白字符（换行、制表等）统一为空格，并去除首尾空格"""
        text = re.sub(r'[\r\n\t\f\v]+', ' ', text)   # 换行/制表 -> 空格
        text = re.sub(r' +', ' ', text)              # 多个空格 -> 单个空格
        return text.strip()

    @classmethod
    def normalize(cls, text: str, to_lower: bool = False) -> str:
        """
        完整的文本规范化流程：
        1. 移除零宽字符
        2. Unicode NFKC 标准化
        3. 全角转半角
        4. 可选小写化
        5. 空白字符归一化
        """
        if not text:
            return text

        # 1. 移除零宽字符
        text = cls.remove_zero_width_chars(text)
        # 2. Unicode 标准化
        text = cls.unicode_normalize(text, 'NFKC')
        # 3. 全角转半角
        text = cls.fullwidth_to_halfwidth(text)
        # 4. 可选小写化（ASCII 字母，中文不受影响）
        if to_lower:
            text = text.lower()
        # 5. 空白字符归一化
        text = cls.normalize_whitespace(text)
        return text


# ---------- 查询预处理器 ----------
class QueryPreprocessor:
    """查询标准化：清洗、规范化、PII掩码"""

    def __init__(self, remove_extra_spaces: bool = True, lowercase: bool = False,
                 normalize_unicode: bool = True, fullwidth_to_halfwidth: bool = True,
                 remove_zero_width: bool = True):
        self.remove_extra_spaces = remove_extra_spaces
        self.lowercase = lowercase
        self.normalize_unicode = normalize_unicode
        self.fullwidth_to_halfwidth = fullwidth_to_halfwidth
        self.remove_zero_width = remove_zero_width

    def normalize(self, text: str) -> str:
        """执行文本规范化（可配置各项步骤）"""
        if not text:
            return ""
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        if self.remove_zero_width:
            text = TextNormalizer.remove_zero_width_chars(text)
        if self.fullwidth_to_halfwidth:
            text = TextNormalizer.fullwidth_to_halfwidth(text)
        if self.lowercase:
            text = text.lower()
        if self.remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text).strip()
        return text

    def clean(self, raw_query: str) -> str:
        """
        对外主方法：先规范化，再 PII 掩码。
        """
        if not raw_query or not isinstance(raw_query, str):
            raise ValueError("Query must be a non-empty string")
        # 1. 规范化
        normalized = self.normalize(raw_query)
        # 2. 掩码 PII
        masked = PIIMask.mask(normalized)
        logger.debug(f"Preprocessed query: {masked[:100]}...")
        return masked