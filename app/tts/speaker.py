import os
from gtts import gTTS
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TTSSpeaker:
    def __init__(self):
        # gTTS supports 'ha', 'yo', 'ig'
        self.supported_langs = ['ha', 'yo', 'ig', 'en']

    def synthesize(self, text: str, language: str, output_path: str) -> Optional[str]:
        lang_code = language.lower()[:2]
        if lang_code not in self.supported_langs:
            lang_code = 'en'
            
        try:
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(output_path)
            return output_path
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            return None
