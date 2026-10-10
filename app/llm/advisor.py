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
        url = f"{settings.natlas_api_url}/v1/chat/completions"
        lang = language[:2].lower()
        system = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
        if context:
            system += f"\n\nUse this agricultural knowledge to answer:\n{context}"
        
        data = {
            "model": "NCAIR1/N-ATLaS",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query}
            ]
        }
        try:
            response = requests.post(url, json=data, timeout=120)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Remote API error: {e}")
            return "Gafara dai, na samu matsala wurin samo amsa. (Sorry, I had trouble getting an answer.)"

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
