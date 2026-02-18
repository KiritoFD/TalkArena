"""
防幻觉机制
输出校验、约束与纠正
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import re


@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    corrected_content: Optional[str] = None
    confidence: float = 1.0


class CharacterConstraintValidator:
    """角色约束验证器"""

    def __init__(self, characters: List[Dict]):
        self.characters = characters
        self.valid_names = {c.get("name", "") for c in characters}
        self.valid_avatars = {c.get("avatar", "") for c in characters}

    def validate_speaker(self, text: str) -> ValidationResult:
        """验证发言者"""
        issues = []

        speaker_match = re.match(r"^([^：:]+)[：:]", text)
        if not speaker_match:
            return ValidationResult(True, confidence=0.7)

        speaker = speaker_match.group(1).strip()

        if speaker not in self.valid_names:
            issues.append(f"未知角色: {speaker}")

        user_indicators = ["你", "用户", "玩家", "我"]
        for indicator in user_indicators:
            if indicator in speaker:
                issues.append(f"禁止替用户发言: {speaker}")

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence=0.9 if len(issues) == 0 else 0.3,
        )

    def validate_content(self, text: str) -> ValidationResult:
        """验证内容"""
        issues = []

        user_patterns = [
            r"你[说想觉得]",
            r"用户[说想觉得]",
            r"你应该",
            r"你可以选择",
        ]

        for pattern in user_patterns:
            if re.search(pattern, text):
                issues.append(f"可能替用户发言: {pattern}")

        if len(text) > 200:
            issues.append(f"内容过长: {len(text)}字符")

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence=0.8 if len(issues) == 0 else 0.4,
        )


class DialogueConstraintValidator:
    """对话约束验证器"""

    FORBIDDEN_PATTERNS = [
        (r"作为AI", "不应出现'作为AI'等元语言"),
        (r"我是一个AI", "不应出现AI自我指涉"),
        (r"我无法|我不能|我不可以", "角色不应自我设限"),
        (r"请问还有什么", "不应出现客服式结尾"),
        (r"希望这有帮助", "不应出现模板化结尾"),
    ]

    REQUIRED_ELEMENTS = {
        "dialect_words": ["昂", "木有", "杠好", "养鱼", "实在", "中", "咱", "您"],
        "toast_actions": ["敬", "喝", "干", "陪", "走一个"],
    }

    def validate(self, text: str) -> ValidationResult:
        """验证对话"""
        issues = []

        for pattern, message in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, text):
                issues.append(message)

        dialect_count = sum(
            1 for word in self.REQUIRED_ELEMENTS["dialect_words"] if word in text
        )
        if dialect_count == 0:
            issues.append("缺少鲁中方言元素")

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence=0.7 + (dialect_count * 0.05),
        )


class ScoreConstraintValidator:
    """分数约束验证器"""

    VALID_RANGE = (0, 100)
    VALID_DELTA_RANGE = (-20, 20)

    def validate_scores(self, scores: Dict[str, float]) -> ValidationResult:
        """验证分数"""
        issues = []

        for key, value in scores.items():
            if not (self.VALID_RANGE[0] <= value <= self.VALID_RANGE[1]):
                issues.append(f"分数 {key}={value} 超出有效范围")

        return ValidationResult(is_valid=len(issues) == 0, issues=issues)

    def validate_dominance(self, dominance: Dict[str, int]) -> ValidationResult:
        """验证气场值"""
        issues = []

        user = dominance.get("user", 50)
        ai = dominance.get("ai", 50)

        if not (0 <= user <= 100):
            issues.append(f"用户气场值 {user} 无效")

        if not (0 <= ai <= 100):
            issues.append(f"AI气场值 {ai} 无效")

        if abs(user + ai - 100) > 1:
            issues.append(f"气场值之和 {user + ai} 不等于100")

        return ValidationResult(is_valid=len(issues) == 0, issues=issues)


class HallucinationCorrector:
    """幻觉纠正器"""

    CORRECTION_RULES = {
        "unknown_speaker": {
            "pattern": r"^[^：:]+[：:]",
            "fix": lambda text, chars: self._fix_unknown_speaker(text, chars),
        },
        "user_proxy": {
            "pattern": r"(你|用户)[应该可以说想]+",
            "fix": lambda text, chars: self._fix_user_proxy(text),
        },
        "too_long": {
            "condition": lambda text: len(text) > 150,
            "fix": lambda text, chars: self._fix_too_long(text),
        },
    }

    def __init__(self, characters: List[Dict]):
        self.characters = characters

    def correct(self, text: str) -> Tuple[str, List[str]]:
        """纠正幻觉"""
        corrections = []
        corrected = text

        for rule_name, rule in self.CORRECTION_RULES.items():
            if "pattern" in rule:
                if re.search(rule["pattern"], corrected):
                    new_text = rule["fix"](corrected, self.characters)
                    if new_text != corrected:
                        corrections.append(f"应用规则: {rule_name}")
                        corrected = new_text
            elif "condition" in rule:
                if rule["condition"](corrected):
                    new_text = rule["fix"](corrected, self.characters)
                    if new_text != corrected:
                        corrections.append(f"应用规则: {rule_name}")
                        corrected = new_text

        return corrected, corrections

    def _fix_unknown_speaker(self, text: str, characters: List[Dict]) -> str:
        """修复未知发言者"""
        match = re.match(r"^([^：:]+)[：:]\s*(.*)", text, re.DOTALL)
        if not match:
            return text

        speaker, content = match.groups()

        valid_names = {c.get("name", "") for c in characters}
        if speaker.strip() in valid_names:
            return text

        if characters:
            default_speaker = characters[0].get("name", "角色")
            return f"{default_speaker}：{content}"

        return content

    def _fix_user_proxy(self, text: str) -> str:
        """修复替用户发言"""
        patterns = [
            (r"你可以说[^。！？]+", ""),
            (r"你应该[^。！？]+", ""),
            (r"你可以选择[^。！？]+", ""),
        ]

        corrected = text
        for pattern, replacement in patterns:
            corrected = re.sub(pattern, replacement, corrected)

        return corrected.strip()

    def _fix_too_long(self, text: str) -> str:
        """修复过长内容"""
        sentences = re.split(r"[。！？]", text)

        selected = []
        total_len = 0
        for sentence in sentences:
            if total_len + len(sentence) <= 80:
                selected.append(sentence)
                total_len += len(sentence)
            else:
                break

        result = "。".join(selected)
        if not result.endswith("。") and selected:
            result += "。"

        return result


class OutputValidator:
    """输出验证器 - 整合所有验证"""

    def __init__(self, characters: List[Dict]):
        self.character_validator = CharacterConstraintValidator(characters)
        self.dialogue_validator = DialogueConstraintValidator()
        self.score_validator = ScoreConstraintValidator()
        self.corrector = HallucinationCorrector(characters)

    def validate_and_correct(self, output: Dict) -> Dict:
        """验证并纠正输出"""
        result = output.copy()
        corrections = []

        if "ai_text" in result:
            text = result["ai_text"]

            speaker_result = self.character_validator.validate_speaker(text)
            content_result = self.character_validator.validate_content(text)
            dialogue_result = self.dialogue_validator.validate(text)

            all_issues = (
                speaker_result.issues + content_result.issues + dialogue_result.issues
            )

            if all_issues:
                corrected_text, text_corrections = self.corrector.correct(text)
                if text_corrections:
                    result["ai_text"] = corrected_text
                    corrections.extend(text_corrections)

        if "scores" in result:
            score_result = self.score_validator.validate_scores(result["scores"])
            if not score_result.is_valid:
                result["scores"] = self._clamp_scores(result["scores"])
                corrections.append("分数已修正到有效范围")

        if "new_dominance" in result:
            dom_result = self.score_validator.validate_dominance(
                result["new_dominance"]
            )
            if not dom_result.is_valid:
                result["new_dominance"] = self._normalize_dominance(
                    result["new_dominance"]
                )
                corrections.append("气场值已修正")

        if corrections:
            result["_corrections"] = corrections

        return result

    def _clamp_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """限制分数范围"""
        return {k: max(0, min(100, v)) for k, v in scores.items()}

    def _normalize_dominance(self, dominance: Dict[str, int]) -> Dict[str, int]:
        """标准化气场值"""
        user = max(10, min(90, dominance.get("user", 50)))
        return {"user": user, "ai": 100 - user}
