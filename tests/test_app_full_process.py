import unittest

from fastapi.testclient import TestClient

import main


class FakeResult:
    def __init__(self, stage, data):
        self.stage = stage
        self.data = data


class FakeThinkResult:
    def __init__(self, content, speaker):
        self.content = content
        self.metadata = {"speaker": speaker}


class FakeDialogueAgent:
    def think(self, ctx):
        speaker = (ctx.get("characters") or [{"name": "面试官"}])[0].get("name", "面试官")
        return FakeThinkResult("欢迎来到模拟面试。", speaker)


class FakeMultiAgent:
    def __init__(self):
        self.agents_list = [FakeDialogueAgent()]


class FakeEngine:
    def __init__(self):
        self.sessions = {}
        self.multi_agent = FakeMultiAgent()

    def start_session(self, scenario_id, characters, scene_name, scene_description, user_info):
        sid = "sess-full-flow"
        self.sessions[sid] = {
            "scenario": {"characters": characters or [{"name": "面试官"}]},
            "turn": 0,
        }
        return sid

    def process_turn(self, session_id, message, multimodal):
        self.sessions[session_id]["turn"] += 1
        turn = self.sessions[session_id]["turn"]
        yield FakeResult(
            "complete",
            {
                "ai_text": f"第{turn}轮回应：收到{message}",
                "speaker": "面试官",
                "judgment": "建议回答更结构化",
                "new_dominance": {"user": 48, "ai": 52},
                "scores": {
                    "emotional_intelligence": 70,
                    "response_quality": 68,
                    "pressure_handling": 66,
                    "cultural_fit": 72,
                },
                "game_over": False,
                "npc_feedback_quality": {"label": "良好", "response_quality": 68, "pressure_handling": 66},
            },
        )

    def get_rescue_suggestion(self, session_id):
        return "先复述问题，再按STAR结构回答。"

    def end_session(self, session_id):
        return {
            "scene_name": "压力面试",
            "medal": "🥈",
            "scores": {"emotional": 72, "reaction": 69, "total": 71},
            "summary": "整体表现稳定，抗压尚可。",
            "suggestion": "下次减少口头禅并增加量化证据。",
        }


class AppFullProcessTests(unittest.TestCase):
    def setUp(self):
        self._old_engine = main.engine
        main.engine = FakeEngine()
        self.client = TestClient(main.app)

    def tearDown(self):
        main.engine = self._old_engine

    def test_dialogue_rescue_summary_full_flow(self):
        start_resp = self.client.post(
            "/api/session/start",
            json={
                "scenario_id": "interview",
                "scene_name": "技术面试",
                "characters": [{"name": "面试官", "role": "技术经理"}, {"name": "HR", "role": "hr"}],
            },
        )
        self.assertEqual(start_resp.status_code, 200)
        start_data = start_resp.json()
        self.assertTrue(start_data["success"])
        sid = start_data["data"]["session_id"]

        chat_resp = self.client.post(
            "/api/chat/send",
            json={
                "session_id": sid,
                "message": "我会用STAR方法来回答",
                "multimodal": {"emotion": {"nervous": 35, "focus": 80}, "voice_level": 42},
            },
        )
        self.assertEqual(chat_resp.status_code, 200)
        chat_data = chat_resp.json()
        self.assertTrue(chat_data["success"])
        self.assertIn("ai_text", chat_data["data"])
        self.assertIn("scores", chat_data["data"])
        self.assertIn("npc_feedback_quality", chat_data["data"])
        self.assertIn("label", chat_data["data"]["npc_feedback_quality"])

        rescue_resp = self.client.post("/api/chat/rescue", json={"session_id": sid, "message": ""})
        self.assertEqual(rescue_resp.status_code, 200)
        rescue_data = rescue_resp.json()
        self.assertTrue(rescue_data["success"])
        self.assertIn("STAR", rescue_data["data"]["suggestion"])

        end_resp = self.client.post("/api/session/end", json={"session_id": sid, "message": ""})
        self.assertEqual(end_resp.status_code, 200)
        end_data = end_resp.json()
        self.assertTrue(end_data["success"])
        self.assertIn("summary", end_data["data"])
        self.assertIn("suggestion", end_data["data"])

    def test_ui_page_contains_talking_head_and_expression_hooks(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        text = page.text

        # 页面可正常打开 + 关键 talking-head/表情状态逻辑存在
        self.assertIn("state-speaking", text)
        self.assertIn("state-reacting", text)
        self.assertIn("function runNonverbalLoop()", text)
        self.assertIn("function applyRenderState(name)", text)
        self.assertIn(".ci.blink .eye", text)
        self.assertIn("@keyframes talkMouth", text)


if __name__ == "__main__":
    unittest.main()
