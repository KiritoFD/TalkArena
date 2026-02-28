import unittest

from core.multimodal_multiagent.speaker_selection import NPCProfile
from core.multimodal_multiagent.contracts import NPCProposal
from core.multimodal_multiagent.orchestrator import MultimodalMultiAgentOrchestrator


class MultiAgentRuntimeTests(unittest.TestCase):
    def test_runtime_returns_speech_for_winner(self):
        rt = MultimodalMultiAgentOrchestrator()
        profiles = {
            "chief": NPCProfile("chief", "chief_interviewer", 0.9),
            "hr": NPCProfile("hr", "hr", 0.7),
            "observer": NPCProfile("observer", "observer", 0.4),
        }
        features = {
            "audio": {"pause_ms": 300, "speech_rate": 4.1, "pitch_jitter": 0.3, "volume_var": 0.3, "pre_speech_motion": 0.7},
            "face": {"brow_furrow": 0.2, "blink_rate_jump": 0.2, "facial_tension": 0.4, "smile": 0.2, "head_forward": 0.5, "gaze_target": "chief"},
            "text": {"mentions": ["chief"], "self_repair": 0.1, "sentiment": 0.0},
        }
        proposals = [
            NPCProposal("chief", True, 0.9, "probe", True, 3000),
            NPCProposal("hr", True, 0.6, "rescue", True, 2800),
            NPCProposal("observer", False, 0.1, "summary", False, 2000),
        ]

        out = rt.run_tick("interview", features, proposals, profiles, {"turn": 2}, ts_ms=1000)
        self.assertIsNotNone(out["speech_instruction"])
        self.assertIn(out["floor_decision"].speaker_id, profiles)
        self.assertEqual(len(out["render_frame"].nonverbals), 3)

    def test_no_valid_proposal(self):
        rt = MultimodalMultiAgentOrchestrator()
        profiles = {"a": NPCProfile("a", "friend", 0.5)}
        features = {"audio": {}, "face": {}, "text": {}}
        proposals = [NPCProposal("a", False, 0.2, "agree", False)]

        out = rt.run_tick("dinner", features, proposals, profiles, {"turn": 0}, ts_ms=100)
        self.assertIsNone(out["speech_instruction"])
        self.assertIsNone(out["floor_decision"].speaker_id)


if __name__ == "__main__":
    unittest.main()
