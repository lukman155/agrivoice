import os
import ssl
import logging
from typing import Optional

# Ensure SSL certificate verification works regardless of machine clock date
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

try:
    from gtts import gTTS
    HAVE_GTTS = True
except ImportError:
    HAVE_GTTS = False

logger = logging.getLogger(__name__)

class TTSSpeaker:
    def __init__(self):
        # Native gTTS voice for Hausa; Nigerian English accent for others
        pass

    def synthesize(self, text: str, language: str, output_path: str) -> Optional[str]:
        if not HAVE_GTTS:
            logger.warning("gTTS not installed. Skipping audio synthesis.")
            return None

        lang_code = language.lower()[:2]
        
        # Extract native closing summary sentence or concise spoken sentence
        import re
        speech_text = text.strip()
        lines = [l.strip() for l in speech_text.split('\n') if l.strip() and not l.startswith('#')]
        if lines and len(lines) > 1 and len(lines[-1]) > 20:
            speech_text = lines[-1]
        elif lines:
            speech_text = lines[0]

        # Clean brackets, colons, and markdown to prevent multi-chunk gTTS overhead
        speech_text = re.sub(r'\(.*?\)', '', speech_text)
        speech_text = speech_text.replace('**', '').replace('*', '').replace(':', ',').replace(';', ',')
        speech_text = re.sub(r'\s+', ' ', speech_text).strip()

        # Normalize Nigerian diacritics to standard phonetics for TTS
        try:
            from natlas.orthography import NigerianOrthographyEngine
            speech_text = NigerianOrthographyEngine.strip_diacritics(speech_text)
        except Exception:
            pass

        if len(speech_text) > 160:
            speech_text = speech_text[:157] + '.'

        try:
            if lang_code == "ha":
                # Native Hausa speech synthesis
                tts = gTTS(text=speech_text, lang="ha", slow=False)
            else:
                # English voice with phonetic clarity
                tts = gTTS(text=speech_text, lang="en", slow=False)

            tts.save(output_path)
            return output_path
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            try:
                # Safe fallback to standard english
                tts = gTTS(text=text[:300], lang="en", slow=False)
                tts.save(output_path)
                return output_path
            except Exception as e2:
                logger.error(f"TTS Fallback failed: {e2}")
                return None
