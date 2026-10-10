import unittest
import numpy as np
from app.voice_pipeline import VoicePipeline

class TestVoicePipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = VoicePipeline()

    def test_mock_transcription(self):
        # Create dummy audio data (1 second of silence)
        dummy_audio = np.zeros(16000, dtype=np.float32)
        
        result = self.pipeline.process_audio(dummy_audio, 16000)
        
        self.assertIn("transcript", result)
        self.assertIn("advice", result)
        self.assertIn("language", result)

if __name__ == '__main__':
    unittest.main()
