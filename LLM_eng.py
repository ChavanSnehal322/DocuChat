
 
import requests
from typing import Any, Dict

class HuggingFaceLLM:
    """
    HuggingFace Router API (2025) – OpenAI-compatible /chat/completions endpoint.
    Works for free models like:
    - mistralai/Mistral-7B-Instruct-v0.2
    - microsoft/Phi-3-mini-4k-instruct
    - google/flan-t5-base
    - meta-llama/Llama-3.1-8B-Instruct
    """
    def __init__(self, hf_token: str, model_name: str, timeout: int = 60):
        self.hf_token = hf_token
        self.model_name = model_name
        self.url = "https://router.huggingface.co/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
        self.timeout = timeout

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.4
        }

        response = requests.post(
            self.url, headers=self.headers, json=payload, timeout=self.timeout
        )

        if response.status_code != 200:
            raise RuntimeError(f"HF Inference error {response.status_code}: {response.text}")

        data = response.json()

        # Extract in OpenAI-style format
        try:
            return data["choices"][0]["message"]["content"]
        except:
            return str(data)
