from typing import Dict, Any, List, Optional
from ..base import ScenarioConfig, CharacterTemplate, DifficultyLevel, ScenarioCategory


class DebateScenarioTemplate:
    template_id = "debate_base"
    template_name = "Debate"
    template_description = "Debate scenario template"

    @classmethod
    def generate_config(
        cls,
        difficulty: Optional[DifficultyLevel] = None,
        topic: str = "ai_impact",
        side: str = "positive",
        **kwargs,
    ) -> ScenarioConfig:
        difficulty = difficulty or DifficultyLevel.MEDIUM

        topics = {
            "ai_impact": {
                "name": "AI对就业的影响",
                "context": "人工智能技术快速发展，对传统工作岗位的冲击",
                "positive_args": [
                    "创造新岗位",
                    "提高效率",
                    "推动创新",
                    "解决老龄化问题",
                ],
                "negative_args": [
                    "取代大量工作",
                    "加剧贫富差距",
                    "引发社会问题",
                    "伦理风险",
                ],
            },
            "remote_work": {
                "name": "远程工作的利弊",
                "context": "后疫情时代远程工作成为新趋势",
                "positive_args": [
                    "提高灵活性",
                    "节省通勤时间",
                    "扩大人才池",
                    "工作生活平衡",
                ],
                "negative_args": [
                    "降低协作效率",
                    "减少社交",
                    "工作生活界限模糊",
                    "管理挑战",
                ],
            },
            "education": {
                "name": "应试教育的改革",
                "context": "中国教育体制的优劣讨论",
                "positive_args": ["保证公平", "培养基础知识", "勤奋精神", "筛选人才"],
                "negative_args": ["扼杀创造力", "唯分数论", "学生压力大", "实践能力弱"],
            },
            "social_media": {
                "name": "社交媒体的影响",
                "context": "社交媒体对社会和个人的影响",
                "positive_args": ["信息传播", "连接世界", "商业机会", "表达自由"],
                "negative_args": ["隐私泄露", "虚假信息", "网络成瘾", "心理健康"],
            },
        }

        chosen = topics.get(topic, topics["ai_impact"])
        arguments = (
            chosen["positive_args"] if side == "positive" else chosen["negative_args"]
        )

        user_side = "正方" if side == "positive" else "反方"

        characters = [
            CharacterTemplate(
                name="主持人",
                role="辩论主持人",
                personality="公正、严谨",
                background="资深辩论赛评委",
                speaking_style="清晰、有节奏",
                possible_questions=["请正方发言", "时间到", "请双方总结"],
                possible_objections=["跑题了", "超时了", "请针对对方观点"],
            ),
            CharacterTemplate(
                name="对方辩友",
                role="对手",
                personality="逻辑清晰、反应快",
                background="专业辩手",
                speaking_style="犀利、有理有据",
                possible_questions=["请问对方...", "我不同意...", "根据数据..."],
                possible_objections=["逻辑不成立", "举例不当", "偷换概念"],
            ),
        ]

        if difficulty in [DifficultyLevel.HARD, DifficultyLevel.EXPERT]:
            characters.append(
                CharacterTemplate(
                    name="观众",
                    role="观众代表",
                    personality="好奇、直接",
                    background="各行各业代表",
                    speaking_style="直接提问",
                    possible_questions=["作为普通人我想问...", "这对我们有什么影响？"],
                    possible_objections=["站着说话不腰疼", "不符合实际"],
                ),
            )

        return ScenarioConfig(
            id=f"debate_{topic}_{side}_{difficulty.value}_{hash(topic) % 1000}",
            name=f"辩论：{chosen['name']}",
            description=f"{user_side}观点：就{chosen['name']}展开辩论",
            category=ScenarioCategory.DEBATE,
            difficulty=difficulty,
            characters=characters,
            user_persona=f"{user_side}辩手",
            context=chosen["context"],
            goals=["清晰表达观点", "有效反驳对方", "逻辑严密", "举例论证"],
            evaluation_criteria=["论点清晰度", "反驳有效性", "逻辑严谨性", "表达能力"],
            max_rounds=5,
            tags=["debate", topic, side],
            metadata={"topic": topic, "side": side, "arguments": arguments},
        )

    @classmethod
    def get_template_info(cls) -> Dict[str, Any]:
        return {
            "template_id": cls.template_id,
            "template_name": cls.template_name,
            "template_description": cls.template_description,
            "supported_topics": [
                "ai_impact",
                "remote_work",
                "education",
                "social_media",
            ],
        }


class DebateScenario(DebateScenarioTemplate):
    template_id = "debate"
    template_name = "Debate Scenario"
    template_description = "辩论场景模拟"
