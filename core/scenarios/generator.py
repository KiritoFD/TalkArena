from typing import Dict, Any, List, Optional
import random
import json
from .base import ScenarioConfig, CharacterTemplate, DifficultyLevel, ScenarioCategory
from .registry import get_registry


class ScenarioGenerator:
    def __init__(self):
        self.registry = get_registry()

    def generate_random_scenario(
        self,
        category: Optional[ScenarioCategory] = None,
        difficulty: Optional[DifficultyLevel] = None,
        **kwargs,
    ) -> ScenarioConfig:
        if category is None:
            category = random.choice(list(ScenarioCategory))

        templates = [
            t
            for t in self.registry.list_templates()
            if t.get("template_id", "").startswith(category.value)
        ]

        if not templates:
            return self._generate_default_scenario(category, difficulty, **kwargs)

        template_class = self.registry.get_template(templates[0]["template_id"])
        if template_class:
            return template_class.generate_config(difficulty=difficulty, **kwargs)

        return self._generate_default_scenario(category, difficulty, **kwargs)

    def _generate_default_scenario(
        self,
        category: ScenarioCategory,
        difficulty: Optional[DifficultyLevel],
        **kwargs,
    ) -> ScenarioConfig:
        difficulty = difficulty or DifficultyLevel.MEDIUM
        char1 = CharacterTemplate(
            name="张三",
            role="主持人",
            personality="友善但严格",
            background="多年主持经验",
            speaking_style="简洁有力",
            possible_questions=["请介绍一下你自己？", "你对这个话题有什么看法？"],
            possible_objections=["这个回答不够深入", "请举例说明"],
        )
        char2 = CharacterTemplate(
            name="李四",
            role="参与者",
            personality="积极主动",
            background="行业新人",
            speaking_style="热情但紧张",
            possible_questions=["你同意他的观点吗？", "能详细说说吗？"],
            possible_objections=["我觉得你说得不对", "这和我了解的不一样"],
        )

        scenario_id = (
            f"{category.value}_{difficulty.value}_{random.randint(1000, 9999)}"
        )

        return ScenarioConfig(
            id=scenario_id,
            name=f"{category.value.title()} Scenario",
            description=f"一个关于{category.value.title()}的模拟场景",
            category=category,
            difficulty=difficulty,
            characters=[char1, char2],
            user_persona="参与者",
            context=f"这是一个{category.value}场景的模拟练习",
            goals=["完成对话", "展现专业能力", "保持礼貌"],
            evaluation_criteria=["回答准确性", "沟通技巧", "应变能力"],
            max_rounds=10,
        )

    def generate_variation(
        self, base_config: ScenarioConfig, modification_level: str = "medium"
    ) -> ScenarioConfig:
        variations = {
            "easy": ["简化背景", "减少角色", "降低目标"],
            "medium": ["调整角色性格", "增加话题深度", "修改目标优先级"],
            "hard": ["增加复杂角色", "引入冲突话题", "设置时间压力"],
        }

        selected_variations = variations.get(modification_level, variations["medium"])

        new_chars = []
        for i, char in enumerate(base_config.characters):
            new_char = CharacterTemplate(
                name=f"{char.name}_{i}",
                role=char.role,
                personality=f"{char.personality}（变体）",
                background=char.background,
                speaking_style=char.speaking_style,
                possible_questions=char.possible_questions,
                possible_objections=char.possible_objections,
            )
            new_chars.append(new_char)

        new_id = f"{base_config.id}_var_{random.randint(100, 999)}"

        return ScenarioConfig(
            id=new_id,
            name=f"{base_config.name} (变体)",
            description=base_config.description,
            category=base_config.category,
            difficulty=base_config.difficulty,
            characters=new_chars,
            user_persona=base_config.user_persona,
            context=base_config.context,
            goals=base_config.goals,
            evaluation_criteria=base_config.evaluation_criteria,
            max_rounds=base_config.max_rounds,
            tags=base_config.tags + ["variation"],
            metadata={**base_config.metadata, "variation_of": base_config.id},
        )

    def export_scenario_as_json(self, config: ScenarioConfig, indent: int = 2) -> str:
        return json.dumps(config.to_dict(), ensure_ascii=False, indent=indent)
