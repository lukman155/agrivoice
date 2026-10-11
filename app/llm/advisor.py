import requests
import json
import logging
from app.config import settings
from app.llm.prompts import build_prompt

logger = logging.getLogger(__name__)

class AgriculturalAdvisor:
    def __init__(self):
        self.backend = settings.llm_backend
        if self.backend == "transformers":
            from transformers import pipeline
            try:
                self.pipe = pipeline(
                    "text-generation",
                    model=settings.llm_model,
                    token=settings.hf_token,
                    device_map="auto"
                )
            except Exception as e:
                logger.error(f"Error loading LLM: {e}")
                self.pipe = None
        else:
            self.pipe = None

    def get_advice(self, query: str, context: str, language: str = "en") -> str:
        prompt = build_prompt(language, context, query)
        
        if self.backend == "ollama":
            return self._call_ollama(prompt)
        elif self.backend == "api":
            return self._call_api(query, context, language)
        elif self.backend == "transformers" and self.pipe:
            return self._call_transformers(prompt)
        else:
            return "Error: LLM backend not available."

    def _call_api(self, query: str, context: str, language: str) -> str:
        """Call remote N-ATLAS API (Colab/Kaggle deployment)."""
        from app.llm.prompts import SYSTEM_PROMPTS
        base_url = (settings.natlas_api_url or "").rstrip('/')
        lang = language[:2].lower()
        
        # Check if an active remote GPU or API endpoint is configured
        is_remote_gpu = bool(base_url and any(h in base_url for h in ["ngrok", "colab", "kaggle", "runpod", "http"]))
        if is_remote_gpu:
            url = f"{base_url}/v1/chat/completions" if not base_url.endswith("/v1") else f"{base_url}/chat/completions"
            system = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
            if context:
                system += f"\n\nUse this agricultural knowledge to answer concisely (under 75 words):\n{context}"
            
            data = {
                "model": "NCAIR1/N-ATLaS",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": query}
                ],
                "max_tokens": 120
            }
            try:
                headers = {
                    "Content-Type": "application/json",
                    "ngrok-skip-browser-warning": "1",
                    "User-Agent": "AgriVoice-Sovereign/1.0"
                }
                logger.info(f"Dispatching query to live N-ATLAS GPU: {url}")
                response = requests.post(url, json=data, headers=headers, timeout=30, verify=False)
                if response.status_code == 200:
                    content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if content and "[API Backend Error" not in content and "[Inference error" not in content:
                        self.last_source = f"Live N-ATLAS 8B GPU ({base_url})"
                        logger.info("Successfully received live response from N-ATLAS GPU.")
                        return content
            except Exception as e:
                logger.warning(f"Remote GPU API unavailable ({e}). Using grounded sovereign RAG advisor fallback.")

        self.last_source = "Grounded Sovereign Extension RAG (Fallback)"
        return self._synthesize_grounded_advice(query, context, lang)

    def _synthesize_grounded_advice(self, query: str, context: str, lang: str) -> str:
        """Synthesize verified agronomic advice from retrieved IITA/NAERLS knowledge."""
        lines = [line.strip().lstrip('-*# ') for line in context.split('\n') if line.strip() and not line.startswith('#')]
        bullets = [l for l in lines if len(l) > 15][:2]
        clean_context = ". ".join(bullets) if bullets else context[:180]
        # Clean any markdown asterisks
        clean_context = clean_context.replace("**", "").replace("*", "")

        if lang == "ha":
            return (
                f"Bisa ga jagoran aikin gona na Najeriya (IITA/NAERLS):\n"
                f"{clean_context}.\n"
                "Koyaushe kiyaye tsabtace gona da amfani da ingantaccen irin shuka da takin zamani."
            )
        elif lang == "yo":
            return (
                f"Gẹgẹbi itọsọna iṣẹ-ogbin Naijiria (IITA/NAERLS):\n"
                f"{clean_context}.\n"
                "Rii daju pe o n lo awọn irugbin ti o ni ilera to dara ati ajile to peye fun oko rẹ."
            )
        elif lang == "ig":
            return (
                f"Dabere na nduzi oru ugbo Naịjirịa (IITA/NAERLS):\n"
                f"{clean_context}.\n"
                "Hụ na ị na-eji mkpụrụ osisi kwesịrị ekwesị ma na-elekọta ala gị nke ọma."
            )
        else:
            return (
                f"According to Nigerian agricultural extension guidelines (IITA/NAERLS):\n"
                f"{clean_context}.\n"
                "Ensure proper weed management, certified seeds, and balanced fertilizer application."
            )

    def _call_ollama(self, prompt: str) -> str:
        url = f"{settings.ollama_url}/api/generate"
        data = {
            "model": "llama3",  # Assuming standard llama3 for local test, or replace with n-atlas if fine-tuned gguf exists
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "repeat_penalty": 1.12
            }
        }
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return "Gafara dai, na samu matsala wurin samo amsa. (Sorry, I had trouble getting an answer.)"

    def _call_transformers(self, prompt: str) -> str:
        try:
            out = self.pipe(
                prompt,
                max_new_tokens=256,
                temperature=0.1,
                repetition_penalty=1.12,
                do_sample=True,
                return_full_text=False
            )
            return out[0]["generated_text"].strip()
        except Exception as e:
            logger.error(f"Transformers generation error: {e}")
            return "Error generating response."
