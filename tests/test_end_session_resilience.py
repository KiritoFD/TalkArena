import unittest

from core.engine import TalkArenaEngine


class DummyLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, prompt, max_new_tokens=200):
        if not self._responses:
            return ""
        return self._responses.pop(0)


class EndSessionResilienceTests(unittest.TestCase):
    def _build_engine(self, llm):
        eng = TalkArenaEngine.__new__(TalkArenaEngine)
        eng.llm = llm
        eng.sessions = {}
        return eng

    def test_end_session_handles_non_numeric_scores(self):
        llm = DummyLLM([
            '{"metrics":{"oily":"N/A","friendliness":"85","logic":null,"humor":"120","respect":"-10"}}',
            '整体不错',
            '{"npc_inner_voice":[],"high_light_suggestion":"继续保持"}',
        ])
        eng = self._build_engine(llm)
        eng.sessions["sid"] = {
            "scene_name": "测试场景",
            "scenario": {"characters": [{"name": "NPC1", "avatar": "👤"}]},
            "dominance": {"user": "70", "ai": None},
            "turn_count": 2,
            "chat_history": [],
        }

        report = eng.end_session("sid")
        self.assertIn("scores", report)
        self.assertEqual(report["scores"]["oily"], 50)
        self.assertEqual(report["scores"]["friendliness"], 85)
        self.assertEqual(report["scores"]["humor"], 100)
        self.assertEqual(report["scores"]["respect"], 0)

    def test_end_session_fallback_when_report_generation_crashes(self):
        class CrashLLM:
            def generate(self, prompt, max_new_tokens=200):
                raise RuntimeError("llm down")

        eng = self._build_engine(CrashLLM())
        eng.sessions["sid"] = {"scene_name": "异常场景", "scenario": {}}

        report = eng.end_session("sid")
        self.assertEqual(report["scene_name"], "异常场景")
        self.assertIn("summary", report)
        self.assertIn("scores", report)


if __name__ == "__main__":
    unittest.main()
