import unittest

from core.multimodal_multiagent.orchestrator import MultimodalMultiAgentOrchestrator
from core.multimodal_multiagent.speaker_selection import NPCProfile


class MultimodalPackageTests(unittest.TestCase):
    def test_new_package_run_tick(self):
        runtime = MultimodalMultiAgentOrchestrator()
        profiles = {
            "chief": NPCProfile("chief", "chief_interviewer", 0.9),
            "hr": NPCProfile("hr", "hr", 0.7),
            "observer": NPCProfile("observer", "observer", 0.4),
        }
        features = {
            "audio": {"pause_ms": 300, "speech_rate": 4.0, "pitch_jitter": 0.3, "volume_var": 0.4, "pre_speech_motion": 0.6},
            "face": {"brow_furrow": 0.3, "blink_rate_jump": 0.2, "facial_tension": 0.4, "smile": 0.2, "head_forward": 0.3, "gaze_target": "chief"},
            "text": {"mentions": ["chief"], "self_repair": 0.1, "sentiment": 0.0},
        }

        result = runtime.run_tick(
            scene="interview",
            features=features,
            proposals=None,
            profiles=profiles,
            context={"turn": 3},
            ts_ms=1000,
        )

        self.assertTrue(result["auto_proposals"])
        self.assertEqual(len(result["render_frame"].nonverbals), 3)
        self.assertIn(result["coordination"].phase, ("opening", "explore", "resolve"))


if __name__ == "__main__":
    unittest.main()
