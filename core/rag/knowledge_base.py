"""
RAG 知识库增强系统
酒桌文化知识检索与生成增强
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import re


@dataclass
class KnowledgeEntry:
    id: str
    category: str
    title: str
    content: str
    keywords: List[str] = field(default_factory=list)
    usage_count: int = 0
    relevance_score: float = 0.0


class ShandongDinnerKnowledgeBase:
    """山东酒桌文化知识库"""

    KNOWLEDGE_DATA = [
        {
            "id": "seat_001",
            "category": "座次礼仪",
            "title": "主陪副陪位置",
            "content": "山东饭局讲究座次：主陪坐主位（正对门），副陪坐主陪对面。主宾在主陪右侧，副宾在主陪左侧。长辈和领导必须坐主位附近。",
            "keywords": ["座次", "主陪", "副陪", "位置", "礼仪"],
        },
        {
            "id": "seat_002",
            "category": "座次礼仪",
            "title": "敬酒顺序",
            "content": "敬酒顺序：先主陪敬全桌，再副陪，然后客人回敬。长辈/领导先敬，晚辈后敬。同辈之间可以互敬。",
            "keywords": ["敬酒", "顺序", "主陪", "副陪", "长辈"],
        },
        {
            "id": "toast_001",
            "category": "劝酒话术",
            "title": "经典劝酒词",
            "content": "山东劝酒常用话术：'感情深一口闷'、'不喝就是看不起我'、'咱山东人实在'、'养鱼呢'（指没喝完）、'我陪一个'（副陪专用）。",
            "keywords": ["劝酒", "话术", "感情深", "山东人"],
        },
        {
            "id": "toast_002",
            "category": "劝酒话术",
            "title": "拒酒技巧",
            "content": "高情商拒酒：'大舅您盛情我领了，但我真不能喝了'、'今天身体不适，以茶代酒'、'改天我专门请您'。要给对方面子。",
            "keywords": ["拒酒", "不喝", "面子", "以茶代酒"],
        },
        {
            "id": "dialect_001",
            "category": "鲁中方言",
            "title": "常用方言词",
            "content": "鲁中饭局常用词：昂（语气词）、木有（没有）、杠好（很好）、养鱼（指酒没喝完）、实在（诚实）、中（行/可以）。",
            "keywords": ["方言", "鲁中", "昂", "木有", "杠好"],
        },
        {
            "id": "dialect_002",
            "category": "鲁中方言",
            "title": "敬酒用语",
            "content": "敬酒时说：'大舅，我敬您一杯'、'给您倒满'、'这杯我干了，您随意'、'咱走一个'。",
            "keywords": ["敬酒", "用语", "随意", "干了"],
        },
        {
            "id": "role_001",
            "category": "角色特点",
            "title": "主陪职责",
            "content": "主陪（通常是长辈或领导）：负责开场、定节奏、带酒。要说开场白，带头敬酒，照顾全桌气氛。敬酒时先干为敬。",
            "keywords": ["主陪", "职责", "开场", "带头"],
        },
        {
            "id": "role_002",
            "category": "角色特点",
            "title": "副陪职责",
            "content": "副陪：配合主陪，负责活跃气氛、劝酒、陪客人。常用'我陪一个'。要让客人喝好但不能让客人难堪。",
            "keywords": ["副陪", "职责", "活跃气氛", "陪酒"],
        },
        {
            "id": "role_003",
            "category": "角色特点",
            "title": "晚辈本分",
            "content": "晚辈在酒桌上：要主动敬酒、眼力见儿要好、不能让长辈等、喝醉了不能失态。说话要谦虚，多用敬语。",
            "keywords": ["晚辈", "本分", "敬酒", "眼力见"],
        },
        {
            "id": "tactic_001",
            "category": "应对策略",
            "title": "转移话题",
            "content": "被劝酒时可以转移话题：夸对方身体好、问对方近况、聊家常、提共同认识的人。化解尴尬又不失礼。",
            "keywords": ["转移", "话题", "化解", "尴尬"],
        },
        {
            "id": "tactic_002",
            "category": "应对策略",
            "title": "以退为进",
            "content": "面对强硬劝酒：先答应再找理由（'好的好的，我缓一下'）、先喝一小口表示诚意、用'来日方长'化解。",
            "keywords": ["以退为进", "缓兵之计", "诚意"],
        },
        {
            "id": "tactic_003",
            "category": "应对策略",
            "title": "借力打力",
            "content": "用对方的话来化解：'您刚才说要照顾我，那今天我就少喝点'、'您身体最重要，咱以茶代酒'。",
            "keywords": ["借力", "化解", "以茶代酒"],
        },
        {
            "id": "etiquette_001",
            "category": "餐桌礼仪",
            "title": "敬酒姿态",
            "content": "敬酒时：双手端杯、杯口低于对方、眼睛看着对方、说敬酒词。长辈敬你要站起来接受。",
            "keywords": ["姿态", "双手", "杯口", "站起来"],
        },
        {
            "id": "etiquette_002",
            "category": "餐桌礼仪",
            "title": "饮酒规矩",
            "content": "山东规矩：主陪带酒要跟、别人敬你要回敬、不能剩酒（养鱼）、不能提前离席。女客人可以少喝但不能不喝。",
            "keywords": ["规矩", "跟酒", "回敬", "养鱼"],
        },
    ]

    def __init__(self):
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.index: Dict[str, List[str]] = {}
        self._build_index()

    def _build_index(self):
        """构建倒排索引"""
        for item in self.KNOWLEDGE_DATA:
            entry = KnowledgeEntry(
                id=item["id"],
                category=item["category"],
                title=item["title"],
                content=item["content"],
                keywords=item.get("keywords", []),
            )
            self.entries[entry.id] = entry

            for keyword in entry.keywords:
                if keyword not in self.index:
                    self.index[keyword] = []
                self.index[keyword].append(entry.id)

        for entry in self.entries.values():
            for word in entry.content:
                if word not in self.index:
                    self.index[word] = []
                if entry.id not in self.index[word]:
                    self.index[word].append(entry.id)

    def retrieve(self, query: str, top_k: int = 3) -> List[KnowledgeEntry]:
        """检索相关知识点"""
        query_words = set(re.findall(r"[\w]+", query.lower()))

        scores: Dict[str, float] = {}

        for word in query_words:
            if word in self.index:
                for entry_id in self.index[word]:
                    if entry_id not in scores:
                        scores[entry_id] = 0
                    scores[entry_id] += 1.0

        for entry_id, entry in self.entries.items():
            for keyword in entry.keywords:
                if keyword in query.lower():
                    if entry_id not in scores:
                        scores[entry_id] = 0
                    scores[entry_id] += 2.0

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[
            :top_k
        ]

        results = []
        for entry_id in sorted_ids:
            entry = self.entries[entry_id]
            entry.relevance_score = scores[entry_id]
            entry.usage_count += 1
            results.append(entry)

        return results

    def augment_prompt(self, query: str, context: Dict = None) -> str:
        """生成增强后的提示词"""
        relevant_entries = self.retrieve(query)

        if not relevant_entries:
            return ""

        knowledge_context = "\n\n".join(
            [f"【{e.category}】{e.title}\n{e.content}" for e in relevant_entries]
        )

        return f"""参考知识库：
{knowledge_context}

请根据以上知识回答问题或生成回应。"""


class RAGEngine:
    """RAG 引擎 - 检索增强生成"""

    def __init__(self, knowledge_base: ShandongDinnerKnowledgeBase = None):
        self.kb = knowledge_base or ShandongDinnerKnowledgeBase()
        self.cache: Dict[str, str] = {}

    def enhance_context(self, user_input: str, context: Dict) -> Dict:
        """增强上下文"""
        knowledge = self.kb.augment_prompt(user_input, context)

        relevant_entries = self.kb.retrieve(user_input)

        category_hints = list(set(e.category for e in relevant_entries))

        return {
            **context,
            "rag_knowledge": knowledge,
            "rag_categories": category_hints,
            "rag_entries": [
                {"title": e.title, "content": e.content, "score": e.relevance_score}
                for e in relevant_entries
            ],
        }

    def get_quick_hint(self, situation: str) -> Optional[str]:
        """获取快速提示"""
        hints = {
            "被劝酒": "建议：可以用'身体不适'、'已经喝多了'、'以茶代酒'等方式礼貌拒绝",
            "气氛冷": "建议：可以敬酒、讲笑话、夸对方、聊共同话题来活跃气氛",
            "座次错": "建议：山东主陪坐主位，副陪对坐，客人坐两侧",
            "不知说什么": "建议：可以敬酒、夸对方、问近况、聊家常",
        }
        return hints.get(situation)

    def build_system_prompt_with_rag(self, base_prompt: str, user_input: str) -> str:
        """构建带RAG增强的系统提示"""
        knowledge = self.kb.augment_prompt(user_input)

        if not knowledge:
            return base_prompt

        return f"""{base_prompt}

【知识库参考】
{knowledge}

请结合以上知识生成更专业、更符合山东酒桌文化的回应。"""
