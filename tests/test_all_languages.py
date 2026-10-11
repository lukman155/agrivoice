import time
import unittest
from app.voice_pipeline import VoicePipeline

class TestMultilingualAgriVoice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = VoicePipeline()

    def test_hausa_voice_pipeline(self):
        t0 = time.time()
        res = self.pipeline.process_text("Yaya zan magance cutar rogo da ke busar da ganyaye?", "ha")
        duration = time.time() - t0
        print(f"\n[HAUSA] Completed in {duration:.2f}s | Audio: {res.get('audio_url')}")
        self.assertEqual(res["language"], "ha")
        self.assertTrue(len(res["advice"]) > 20)
        self.assertIsNotNone(res["audio_url"])

    def test_yoruba_voice_pipeline(self):
        t0 = time.time()
        res = self.pipeline.process_text("Kini itoju to ye fun kokoro to n ba agbado je?", "yo")
        duration = time.time() - t0
        print(f"\n[YORUBA] Completed in {duration:.2f}s | Audio: {res.get('audio_url')}")
        self.assertEqual(res["language"], "yo")
        self.assertTrue(len(res["advice"]) > 20)
        self.assertIsNotNone(res["audio_url"])

    def test_igbo_voice_pipeline(self):
        t0 = time.time()
        res = self.pipeline.process_text("Kedu ka m ga-esi chekwaa ji n'oba ka o ghara ire ure?", "ig")
        duration = time.time() - t0
        print(f"\n[IGBO] Completed in {duration:.2f}s | Audio: {res.get('audio_url')}")
        self.assertEqual(res["language"], "ig")
        self.assertTrue(len(res["advice"]) > 20)
        self.assertIsNotNone(res["audio_url"])

    def test_english_voice_pipeline(self):
        t0 = time.time()
        res = self.pipeline.process_text("What is the best fertilizer timing for maize farming?", "en")
        duration = time.time() - t0
        print(f"\n[ENGLISH] Completed in {duration:.2f}s | Audio: {res.get('audio_url')}")
        self.assertEqual(res["language"], "en")
        self.assertTrue(len(res["advice"]) > 20)
        self.assertIsNotNone(res["audio_url"])

if __name__ == "__main__":
    unittest.main()
