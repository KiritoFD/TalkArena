"""
Agent 决策引擎
任务拆解、多步规划与执行
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import re


class TaskType(Enum):
    DIALOGUE = "dialogue"
    EVALUATION = "evaluation"
    RESCUE = "rescue"
    ANALYSIS = "analysis"
    REPORT = "report"


class TaskPriority(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class Task:
    id: str
    type: TaskType
    description: str
    priority: TaskPriority
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Any = None
    sub_tasks: List["Task"] = field(default_factory=list)


@dataclass
class Plan:
    id: str
    goal: str
    tasks: List[Task] = field(default_factory=list)
    current_step: int = 0
    context: Dict = field(default_factory=dict)


class TaskDecomposer:
    """任务拆解器 - 将复杂任务分解为子任务"""

    TASK_TEMPLATES = {
        "handle_turn": [
            {"type": "analysis", "desc": "分析用户输入意图和情感"},
            {"type": "retrieval", "desc": "检索相关知识库内容"},
            {"type": "generation", "desc": "生成AI角色回应"},
            {"type": "evaluation", "desc": "评估用户表现"},
            {"type": "update", "desc": "更新游戏状态"},
        ],
        "generate_rescue": [
            {"type": "context", "desc": "获取当前上下文"},
            {"type": "analysis", "desc": "分析用户处境"},
            {"type": "strategy", "desc": "确定救场策略"},
            {"type": "generation", "desc": "生成建议"},
        ],
        "end_game": [
            {"type": "summary", "desc": "生成对话总结"},
            {"type": "analysis", "desc": "分析整体表现"},
            {"type": "report", "desc": "生成报告"},
        ],
    }

    def decompose(self, task_type: str, context: Dict) -> List[Task]:
        """拆解任务"""
        template = self.TASK_TEMPLATES.get(task_type, [])

        tasks = []
        for i, step in enumerate(template):
            task = Task(
                id=f"{task_type}_{i}",
                type=TaskType.ANALYSIS,
                description=step["desc"],
                priority=TaskPriority.HIGH if i == 0 else TaskPriority.MEDIUM,
                dependencies=[f"{task_type}_{i - 1}"] if i > 0 else [],
            )
            tasks.append(task)

        return tasks


class Planner:
    """规划器 - 多步规划"""

    def __init__(self):
        self.decomposer = TaskDecomposer()
        self.plans: Dict[str, Plan] = {}

    def create_plan(self, goal: str, context: Dict) -> Plan:
        """创建执行计划"""
        task_type = self._identify_task_type(goal)
        tasks = self.decomposer.decompose(task_type, context)

        plan = Plan(
            id=f"plan_{len(self.plans)}", goal=goal, tasks=tasks, context=context
        )

        self.plans[plan.id] = plan
        return plan

    def _identify_task_type(self, goal: str) -> str:
        """识别任务类型"""
        goal_lower = goal.lower()

        if "救场" in goal or "建议" in goal:
            return "generate_rescue"
        elif "结束" in goal or "报告" in goal:
            return "end_game"
        else:
            return "handle_turn"

    def get_next_task(self, plan_id: str) -> Optional[Task]:
        """获取下一个待执行任务"""
        plan = self.plans.get(plan_id)
        if not plan:
            return None

        for task in plan.tasks:
            if task.status == "pending":
                deps_complete = all(
                    self._is_task_complete(plan, dep_id) for dep_id in task.dependencies
                )
                if deps_complete:
                    return task

        return None

    def _is_task_complete(self, plan: Plan, task_id: str) -> bool:
        """检查任务是否完成"""
        for task in plan.tasks:
            if task.id == task_id:
                return task.status == "completed"
        return False

    def update_task_status(
        self, plan_id: str, task_id: str, status: str, result: Any = None
    ):
        """更新任务状态"""
        plan = self.plans.get(plan_id)
        if not plan:
            return

        for task in plan.tasks:
            if task.id == task_id:
                task.status = status
                if result:
                    task.result = result
                break


class DecisionEngine:
    """决策引擎 - 整合分析与决策"""

    DECISION_RULES = {
        "response_strategy": {
            "conditions": [
                {"if": {"dominance.user": "<30"}, "then": "defensive"},
                {"if": {"dominance.user": ">70"}, "then": "aggressive"},
                {"if": {"turn_count": ">5"}, "then": "pressure"},
                {"else": "balanced"},
            ]
        },
        "evaluation_weight": {
            "conditions": [
                {
                    "if": {"multimodal.available": True},
                    "then": {"text": 0.4, "emotion": 0.3, "voice": 0.3},
                },
                {"else": {"text": 0.6, "emotion": 0.2, "voice": 0.2}},
            ]
        },
        "game_over": {
            "conditions": [
                {"if": {"dominance.user": "<=10"}, "then": True},
                {"if": {"dominance.user": ">=90"}, "then": True},
                {"if": {"turn_count": ">=20"}, "then": True},
                {"else": False},
            ]
        },
    }

    def __init__(self):
        self.planner = Planner()

    def analyze_input(self, user_input: str, context: Dict) -> Dict:
        """分析用户输入"""
        analysis = {
            "intent": self._detect_intent(user_input),
            "sentiment": self._detect_sentiment(user_input),
            "topics": self._extract_topics(user_input),
            "entities": self._extract_entities(user_input),
            "strategies": self._suggest_strategies(user_input, context),
        }
        return analysis

    def _detect_intent(self, text: str) -> str:
        """检测意图"""
        text = text.lower()

        if any(w in text for w in ["不喝", "不能喝", "喝不了", "不行"]):
            return "refuse"
        if any(w in text for w in ["好", "行", "可以", "没问题", "来"]):
            return "accept"
        if any(w in text for w in ["敬", "干", "走一个", "陪"]):
            return "toast"
        if any(w in text for w in ["谢谢", "感谢", "麻烦"]):
            return "thank"
        if "?" in text or "？" in text:
            return "question"

        return "neutral"

    def _detect_sentiment(self, text: str) -> Dict:
        """检测情感"""
        positive_words = ["好", "行", "高兴", "开心", "喜欢", "谢谢", "感谢"]
        negative_words = ["不", "没", "烦", "累", "难受", "不行", "不要"]

        positive_count = sum(1 for w in positive_words if w in text)
        negative_count = sum(1 for w in negative_words if w in text)

        if positive_count > negative_count:
            return {"label": "positive", "score": 0.7}
        elif negative_count > positive_count:
            return {"label": "negative", "score": 0.6}
        return {"label": "neutral", "score": 0.5}

    def _extract_topics(self, text: str) -> List[str]:
        """提取主题"""
        topics = []

        topic_keywords = {
            "喝酒": ["喝", "酒", "干", "敬", "杯"],
            "工作": ["工作", "单位", "公司", "老板"],
            "家庭": ["家里", "孩子", "老婆", "老公"],
            "健康": ["身体", "病", "医生", "药"],
            "钱": ["钱", "工资", "收入", "花"],
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in text for kw in keywords):
                topics.append(topic)

        return topics

    def _extract_entities(self, text: str) -> List[Dict]:
        """提取实体"""
        entities = []

        relations = ["大舅", "大妗子", "表哥", "二叔", "舅妈", "叔叔", "阿姨", "姑姑"]
        for rel in relations:
            if rel in text:
                entities.append({"type": "relation", "value": rel})

        numbers = re.findall(r"\d+", text)
        for num in numbers:
            entities.append({"type": "number", "value": num})

        return entities

    def _suggest_strategies(self, text: str, context: Dict) -> List[str]:
        """建议策略"""
        strategies = []

        dominance = context.get("dominance", {}).get("user", 50)

        if dominance < 30:
            strategies.append("以退为进")
            strategies.append("借力打力")
        elif dominance > 70:
            strategies.append("乘胜追击")
            strategies.append("保持节奏")
        else:
            strategies.append("稳扎稳打")
            strategies.append("随机应变")

        intent = self._detect_intent(text)
        if intent == "refuse":
            strategies.append("委婉拒绝")
        elif intent == "accept":
            strategies.append("顺势而为")

        return strategies

    def make_decision(self, context: Dict) -> Dict:
        """做出决策"""
        decisions = {}

        for rule_name, rule_config in self.DECISION_RULES.items():
            for condition in rule_config["conditions"]:
                if self._evaluate_condition(condition.get("if", {}), context):
                    decisions[rule_name] = condition["then"]
                    break
            else:
                if "else" in rule_config["conditions"][-1]:
                    decisions[rule_name] = rule_config["conditions"][-1]["else"]

        return decisions

    def _evaluate_condition(self, condition: Dict, context: Dict) -> bool:
        """评估条件"""
        for key, expected in condition.items():
            actual = self._get_nested_value(context, key)

            if isinstance(expected, str):
                if expected.startswith("<="):
                    return actual <= float(expected[2:])
                elif expected.startswith(">="):
                    return actual >= float(expected[2:])
                elif expected.startswith("<"):
                    return actual < float(expected[1:])
                elif expected.startswith(">"):
                    return actual > float(expected[1:])
                else:
                    return actual == expected
            elif isinstance(expected, bool):
                return actual == expected
            elif isinstance(expected, (int, float)):
                return actual == expected

        return False

    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """获取嵌套值"""
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, 0)
            else:
                return 0
        return value

    def create_execution_plan(self, goal: str, context: Dict) -> Plan:
        """创建执行计划"""
        return self.planner.create_plan(goal, context)

    def execute_step(self, plan: Plan, step_fn) -> Tuple[bool, Any]:
        """执行一步"""
        task = self.planner.get_next_task(plan.id)
        if not task:
            return True, None

        try:
            result = step_fn(task, plan.context)
            self.planner.update_task_status(plan.id, task.id, "completed", result)
            return False, result
        except Exception as e:
            self.planner.update_task_status(plan.id, task.id, "failed", str(e))
            return True, str(e)
