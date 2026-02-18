from .templates.dinner import DinnerScenario
from .templates.interview import InterviewScenario
from .templates.debate import DebateScenario
from .registry import register_scenario

register_scenario(DinnerScenario)
register_scenario(InterviewScenario)
register_scenario(DebateScenario)
