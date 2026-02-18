from typing import Dict, Any, List, Optional
from ..base import ScenarioConfig, CharacterTemplate, DifficultyLevel, ScenarioCategory


class DinnerScenarioTemplate:
    template_id = "dinner_base"
    template_name = "Dinner"
    template_description = "Shandong dinner conversation scenario template"

    @classmethod
    def generate_config(
        cls,
        difficulty: Optional[DifficultyLevel] = None,
        specific_scenario: str = "family_dinner",
        **kwargs,
    ) -> ScenarioConfig:
        difficulty = difficulty or DifficultyLevel.MEDIUM

        scenarios = {
            "family_dinner": {
                "name": "家庭聚餐",
                "description": "山东家庭聚餐场景，学习如何在酒桌上得体应对",
                "context": "春节团圆饭，长辈、亲戚、晚辈齐聚一堂",
                "user_persona": "刚工作的年轻人",
                "goals": ["尊敬长辈", "巧妙敬酒", "应对催婚", "化解尴尬"],
                "evaluation_criteria": [
                    "礼仪表现",
                    "应变能力",
                    "语言得体度",
                    "情商展现",
                ],
            },
            "business_dinner": {
                "name": "商务宴请",
                "description": "商务场合的酒桌文化，学习职场应酬",
                "context": "与合作伙伴的商务晚餐，需要谈成合作",
                "user_persona": "项目经理",
                "goals": ["建立关系", "谈成合作", "不失礼节", "掌握分寸"],
                "evaluation_criteria": ["商务礼仪", "沟通技巧", "谈判能力", "酒桌文化"],
            },
            "senior_meeting": {
                "name": "领导会面",
                "description": "与领导或上级的饭局，学习职场生存",
                "context": "部门聚餐，需要在领导面前表现",
                "user_persona": "新入职员工",
                "goals": ["印象管理", "适度表现", "倾听学习", "避免失误"],
                "evaluation_criteria": ["职场礼仪", "表现分寸", "印象管理", "学习态度"],
            },
        }

        chosen = scenarios.get(specific_scenario, scenarios["family_dinner"])

        characters = [
            CharacterTemplate(
                name="父亲",
                role="家中长辈",
                personality="严肃但关爱子女",
                background="传统山东人，重视礼仪",
                speaking_style="言简意赅，常常用典故",
                possible_questions=["工作怎么样？", "有没有对象？", "工资多少？"],
                possible_objections=[
                    "怎么还不结婚？",
                    "看看别人家孩子...",
                    "你怎么不懂事？",
                ],
            ),
            CharacterTemplate(
                name="叔叔",
                role="亲戚",
                personality="热情好酒，爱开玩笑",
                background="经常应酬，酒量好",
                speaking_style="豪爽，直接",
                possible_questions=["来，喝一个！", "你能喝多少？", "这酒怎么样？"],
                possible_objections=["不喝不给面子！", "养鱼呢？"],
            ),
            CharacterTemplate(
                name="姑姑",
                role="亲戚",
                personality="热心肠，爱操心",
                background="退休教师，喜欢关心晚辈",
                speaking_style="啰嗦但善意",
                possible_questions=[
                    "找对象了吗？",
                    "什么时候买房？",
                    "怎么不考公务员？",
                ],
                possible_objections=["眼光别太高", "老大不小了"],
            ),
        ]

        return ScenarioConfig(
            id=f"dinner_{specific_scenario}_{difficulty.value}_{hash(specific_scenario) % 1000}",
            name=chosen["name"],
            description=chosen["description"],
            category=ScenarioCategory.DINNER,
            difficulty=difficulty,
            characters=characters,
            user_persona=chosen["user_persona"],
            context=chosen["context"],
            goals=chosen["goals"],
            evaluation_criteria=chosen["evaluation_criteria"],
            max_rounds=8,
            tags=["dinner", "shandong", "etiquette", specific_scenario],
            metadata={"specific_scenario": specific_scenario},
        )

    @classmethod
    def get_template_info(cls) -> Dict[str, Any]:
        return {
            "template_id": cls.template_id,
            "template_name": cls.template_name,
            "template_description": cls.template_description,
            "supported_scenarios": [
                "family_dinner",
                "business_dinner",
                "senior_meeting",
            ],
        }


class DinnerScenario(DinnerScenarioTemplate):
    template_id = "dinner"
    template_name = "Dinner Scenario"
    template_description = "山东饭桌文化场景"
