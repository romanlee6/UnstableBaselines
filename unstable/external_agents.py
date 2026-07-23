import os
import time
from typing import Optional
from urllib.parse import urlparse


STANDARD_GAME_PROMPT = (
    "You are a competitive game player. Make sure you read the game instructions "
    "carefully, and always follow the required format."
)


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def azure_ai_endpoint() -> str:
    """Return an OpenAI-compatible Azure Foundry endpoint without exposing secrets."""
    endpoint = _first_env("AZURE_AI_ENDPOINT", "AZURE_OPENAI_ENDPOINT", "AZURE_FOUNDRY_ENDPOINT")
    if not endpoint:
        endpoint = _first_env("AZURE_AI_RESOURCE", "AZURE_OPENAI_RESOURCE", "ANTHROPIC_FOUNDRY_RESOURCE")
    if not endpoint:
        raise ValueError(
            "Azure AI endpoint not found. Set AZURE_AI_ENDPOINT (preferred), "
            "AZURE_OPENAI_ENDPOINT, or ANTHROPIC_FOUNDRY_RESOURCE."
        )

    endpoint = endpoint.strip().rstrip("/")
    if "://" not in endpoint:
        endpoint = f"https://{endpoint}.services.ai.azure.com"

    parsed = urlparse(endpoint)
    if parsed.path in ("", "/") and parsed.hostname and (
        parsed.hostname.endswith(".services.ai.azure.com")
        or parsed.hostname.endswith(".openai.azure.com")
    ):
        endpoint += "/openai/v1"
    return endpoint.rstrip("/") + "/"


def azure_ai_api_key() -> str:
    key = _first_env(
        "AZURE_AI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_INFERENCE_CREDENTIAL",
        "ANTHROPIC_FOUNDRY_API_KEY",
    )
    if not key:
        raise ValueError(
            "Azure AI API key not found. Set AZURE_AI_API_KEY (preferred), "
            "AZURE_OPENAI_API_KEY, AZURE_INFERENCE_CREDENTIAL, or "
            "ANTHROPIC_FOUNDRY_API_KEY."
        )
    return key


class AzureAIAgent:
    """Synchronous fixed opponent backed by Azure AI Foundry's OpenAI/v1 API."""

    def __init__(
        self,
        model_name: str,
        system_prompt: str = STANDARD_GAME_PROMPT,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        retries: int = 5,
    ):
        from openai import OpenAI

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens if max_tokens is not None else int(os.environ.get("UB_EVAL_MAX_TOKENS", "4096"))
        self.temperature = temperature if temperature is not None else float(os.environ.get("UB_EVAL_TEMPERATURE", "0.0"))
        self.retries = retries
        key = azure_ai_api_key()
        # Azure's OpenAI/v1 endpoints accept API-key authentication. Supplying
        # api-key explicitly also supports Foundry endpoints that do not accept
        # the OpenAI SDK's default Bearer header.
        self.client = OpenAI(
            base_url=azure_ai_endpoint(),
            api_key=key,
            default_headers={"api-key": key},
        )

    def __call__(self, observation: str) -> str:
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string, got {type(observation)}")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": observation},
        ]
        last_error = None
        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("Azure AI returned an empty completion")
                return content.strip()
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(2 ** attempt, 30))
        raise last_error


def build_external_agent(provider: str, model_name: str):
    normalized = provider.lower().replace("-", "_")
    if normalized == "openrouter":
        import textarena as ta

        return ta.agents.OpenRouterAgent(model_name)
    if normalized in {"azure", "azure_ai", "azure_foundry"}:
        return AzureAIAgent(model_name)
    raise ValueError(f"Unsupported external evaluation provider: {provider!r}")
