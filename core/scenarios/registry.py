from typing import Dict, List, Optional, Type
from .base import ScenarioConfig, ScenarioTemplate, ScenarioCategory


class ScenarioRegistry:
    _instance: Optional["ScenarioRegistry"] = None
    _templates: Dict[str, Type[ScenarioTemplate]] = {}
    _configs: Dict[str, ScenarioConfig] = {}
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._templates = {}
            self._configs = {}
            self._initialized = True
            self._register_builtins()

    def _register_builtins(self):
        try:
            from .templates.dinner import DinnerScenario
            from .templates.interview import InterviewScenario
            from .templates.debate import DebateScenario

            self.register_template(DinnerScenario)
            self.register_template(InterviewScenario)
            self.register_template(DebateScenario)
        except ImportError:
            pass

    @classmethod
    def get_instance(cls) -> "ScenarioRegistry":
        if cls._instance is None:
            cls._instance = ScenarioRegistry()
        return cls._instance

    def register_template(self, template: Type[ScenarioTemplate]) -> None:
        self._templates[template.template_id] = template

    def register_config(self, config: ScenarioConfig) -> None:
        self._configs[config.id] = config

    def get_template(self, template_id: str) -> Optional[Type[ScenarioTemplate]]:
        return self._templates.get(template_id)

    def get_config(self, config_id: str) -> Optional[ScenarioConfig]:
        return self._configs.get(config_id)

    def list_templates(self) -> List[Dict[str, str]]:
        return [t.get_template_info() for t in self._templates.values()]

    def list_configs(self) -> List[ScenarioConfig]:
        return list(self._configs.values())

    def list_configs_by_category(
        self, category: ScenarioCategory
    ) -> List[ScenarioConfig]:
        return [c for c in self._configs.values() if c.category == category]

    def list_config_ids(self) -> List[str]:
        return list(self._configs.keys())

    def clear(self) -> None:
        self._templates.clear()
        self._configs.clear()


def register_scenario(template: Type[ScenarioTemplate]):
    registry = ScenarioRegistry.get_instance()
    registry.register_template(template)
    return template


def get_registry() -> ScenarioRegistry:
    return ScenarioRegistry.get_instance()


def list_available_scenarios() -> List[str]:
    registry = ScenarioRegistry.get_instance()
    return registry.list_config_ids()
