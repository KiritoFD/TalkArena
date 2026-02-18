from typing import Dict, Any, List, Optional
from ..base import ScenarioConfig, CharacterTemplate, DifficultyLevel, ScenarioCategory


class InterviewScenarioTemplate:
    template_id = "interview_base"
    template_name = "Interview"
    template_description = "Job interview scenario template"

    @classmethod
    def generate_config(
        cls,
        difficulty: Optional[DifficultyLevel] = None,
        interview_type: str = "technical",
        **kwargs,
    ) -> ScenarioConfig:
        difficulty = difficulty or DifficultyLevel.MEDIUM

        interview_types = {
            "technical": {
                "name": "技术面试",
                "description": "技术岗位面试，展示专业能力",
                "context": "知名科技公司的技术岗位面试",
                "user_persona": "求职者",
                "goals": ["展示技术能力", "清晰表达思路", "应对难题", "展现学习能力"],
                "evaluation_criteria": [
                    "技术深度",
                    "表达清晰度",
                    "问题解决能力",
                    "学习潜力",
                ],
            },
            "hr": {
                "name": "HR面试",
                "description": "人力资源面试，了解求职动机和软实力",
                "context": "公司HR进行的综合素质面试",
                "user_persona": "求职者",
                "goals": ["表达求职动机", "展现价值观", "回答行为问题", "展现团队合作"],
                "evaluation_criteria": [
                    "沟通表达",
                    "价值观匹配",
                    "自我认知",
                    "职业规划",
                ],
            },
            "behavioral": {
                "name": "行为面试",
                "description": "行为面试，通过STAR法则展示过往经历",
                "context": "使用STAR法则的行为面试",
                "user_persona": "求职者",
                "goals": ["用STAR法则讲故事", "展示真实案例", "反思成长", "展现软技能"],
                "evaluation_criteria": [
                    "故事结构",
                    "案例真实性",
                    "反思深度",
                    "软技能展示",
                ],
            },
            "group": {
                "name": "群面",
                "description": "无领导小组讨论，展现团队协作",
                "context": "群面形式的小组讨论",
                "user_persona": "求职者",
                "goals": ["积极发言", "推动讨论", "协调分歧", "展现领导力"],
                "evaluation_criteria": ["参与度", "协调能力", "观点质量", "领导潜质"],
            },
        }

        chosen = interview_types.get(interview_type, interview_types["technical"])

        base_chars = [
            CharacterTemplate(
                name="面试官",
                role="主面试官",
                personality="专业、观察细致",
                background="资深HR或技术经理",
                speaking_style="专业但友善",
                possible_questions=[
                    "请介绍一下你自己？",
                    "为什么想来我们公司？",
                    "你的优点和缺点是什么？",
                ],
                possible_objections=["这不够具体", "能举个具体例子吗？", "还有呢？"],
            ),
        ]

        if interview_type == "group":
            base_chars.extend(
                [
                    CharacterTemplate(
                        name="求职者A",
                        role="竞争者",
                        personality="强势，爱表现",
                        background="有经验的候选人",
                        speaking_style="直接，经常打断他人",
                        possible_questions=["我有个想法...", "我觉得应该..."],
                        possible_objections=["我不同意...", "我的方案更好"],
                    ),
                    CharacterTemplate(
                        name="求职者B",
                        role="竞争者",
                        personality="沉默寡言",
                        background="应届生",
                        speaking_style="简短",
                        possible_questions=[],
                        possible_objections=[],
                    ),
                ]
            )

        return ScenarioConfig(
            id=f"interview_{interview_type}_{difficulty.value}_{hash(interview_type) % 1000}",
            name=chosen["name"],
            description=chosen["description"],
            category=ScenarioCategory.INTERVIEW,
            difficulty=difficulty,
            characters=base_chars,
            user_persona=chosen["user_persona"],
            context=chosen["context"],
            goals=chosen["goals"],
            evaluation_criteria=chosen["evaluation_criteria"],
            max_rounds=6,
            tags=["interview", interview_type, "job_search"],
            metadata={"interview_type": interview_type},
        )

    @classmethod
    def get_template_info(cls) -> Dict[str, Any]:
        return {
            "template_id": cls.template_id,
            "template_name": cls.template_name,
            "template_description": cls.template_description,
            "supported_types": ["technical", "hr", "behavioral", "group"],
        }


class InterviewScenario(InterviewScenarioTemplate):
    template_id = "interview"
    template_name = "Interview Scenario"
    template_description = "面试场景模拟"
