from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json
import os


class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class ScenarioCategory(Enum):
    DINNER = "dinner"
    INTERVIEW = "interview"
    DEBATE = "debate"
    NEGOTIATION = "negotiation"
    SOCIAL = "social"
    CUSTOM = "custom"


@dataclass
class CharacterTemplate:
    name: str
    role: str
    personality: str
    background: str
    speaking_style: str
    possible_questions: List[str] = field(default_factory=list)
    possible_objections: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "personality": self.personality,
            "background": self.background,
            "speaking_style": self.speaking_style,
            "possible_questions": self.possible_questions,
            "possible_objections": self.possible_objections,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterTemplate":
        return cls(
            name=data["name"],
            role=data["role"],
            personality=data["personality"],
            background=data["background"],
            speaking_style=data["speaking_style"],
            possible_questions=data.get("possible_questions", []),
            possible_objections=data.get("possible_objections", []),
        )


@dataclass
class ScenarioConfig:
    id: str
    name: str
    description: str
    category: ScenarioCategory
    difficulty: DifficultyLevel
    characters: List[CharacterTemplate]
    user_persona: str
    context: str
    goals: List[str]
    evaluation_criteria: List[str]
    time_limit: Optional[int] = None
    max_rounds: int = 10
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "characters": [c.to_dict() for c in self.characters],
            "user_persona": self.user_persona,
            "context": self.context,
            "goals": self.goals,
            "evaluation_criteria": self.evaluation_criteria,
            "time_limit": self.time_limit,
            "max_rounds": self.max_rounds,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    def save_to_file(self, directory: str = "scenarios") -> str:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{self.id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return file_path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioConfig":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            category=ScenarioCategory(data["category"]),
            difficulty=DifficultyLevel(data["difficulty"]),
            characters=[CharacterTemplate.from_dict(c) for c in data["characters"]],
            user_persona=data["user_persona"],
            context=data["context"],
            goals=data["goals"],
            evaluation_criteria=data["evaluation_criteria"],
            time_limit=data.get("time_limit"),
            max_rounds=data.get("max_rounds", 10),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def load_from_file(cls, file_path: str) -> "ScenarioConfig":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class ScenarioTemplate:
    template_id: str = "base"
    template_name: str = "Base Scenario"
    template_description: str = "Base scenario template"

    @classmethod
    def generate_config(cls, **kwargs) -> ScenarioConfig:
        raise NotImplementedError("Subclasses must implement generate_config")

    @classmethod
    def get_template_info(cls) -> Dict[str, Any]:
        return {
            "template_id": cls.template_id,
            "template_name": cls.template_name,
            "template_description": cls.template_description,
        }
