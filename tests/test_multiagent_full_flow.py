import unittest

from core.multimodal_multiagent.speaker_selection import NPCProfile
from core.multimodal_multiagent.orchestrator import MultimodalMultiAgentOrchestrator


class MultiAgentFullFlowTests(unittest.TestCase):
    def setUp(self):
        self.rt = MultimodalMultiAgentOrchestrator()

    def test_interview_full_flow_with_auto_proposals(self):
        profiles = {
            "chief": NPCProfile("chief", "chief_interviewer", 0.9),
            "tech": NPCProfile("tech", "tech_interviewer", 0.78),
            "hr": NPCProfile("hr", "hr", 0.73),
            "observer": NPCProfile("observer", "observer", 0.35),
        }
        features = {
            "audio": {
                "pause_ms": 900,
                "speech_rate": 2.8,
                "pitch_jitter": 0.62,
                "volume_var": 0.56,
                "pre_speech_motion": 0.35,
            },
            "face": {
                "brow_furrow": 0.64,
                "blink_rate_jump": 0.52,
                "facial_tension": 0.61,
                "smile": 0.1,
                "head_forward": 0.2,
                "gaze_target": "hr",
            },
            "text": {"mentions": ["hr"], "self_repair": 0.66, "sentiment": -0.4},
        }

        out = self.rt.run_tick(
            scene="interview",
            features=features,
            proposals=None,
            profiles=profiles,
            context={"turn": 4},
            ts_ms=1200,
        )

        self.assertTrue(out["auto_proposals"])
        self.assertEqual(len(out["proposals"]), 4)
        self.assertEqual(len(out["render_frame"].nonverbals), 4)
        self.assertIn(out["coordination"].phase, ("opening", "explore", "resolve"))
        self.assertIsNotNone(out["speech_instruction"])
        self.assertEqual(out["speech_instruction"].npc_id, out["floor_decision"].speaker_id)
        self.assertIn("phase=", out["speech_instruction"].text_prompt)

    def test_scene_coordination_changes_with_turns(self):
        profiles = {
            "host": NPCProfile("host", "host", 0.75),
            "friend": NPCProfile("friend", "friend", 0.58),
            "elder": NPCProfile("elder", "elder", 0.65),
        }
        features = {
            "audio": {"pause_ms": 1300, "speech_rate": 3.1, "pitch_jitter": 0.48, "volume_var": 0.44, "pre_speech_motion": 0.25},
            "face": {"brow_furrow": 0.6, "blink_rate_jump": 0.45, "facial_tension": 0.52, "smile": 0.2, "head_forward": 0.2, "gaze_target": "host"},
            "text": {"mentions": ["host"], "self_repair": 0.51, "sentiment": -0.2},
        }

        out_early = self.rt.run_tick("dinner", features, None, profiles, {"turn": 0}, ts_ms=100)
        out_late = self.rt.run_tick("dinner", features, None, profiles, {"turn": 7}, ts_ms=3300)

        self.assertEqual(out_early["coordination"].phase, "opening")
        self.assertEqual(out_late["coordination"].phase, "resolve")
        self.assertTrue(out_late["coordination"].should_soften)

        speaking_count = sum(1 for n in out_late["render_frame"].nonverbals if n.state == "speaking")
        self.assertEqual(speaking_count, 1)


if __name__ == "__main__":
    unittest.main()
