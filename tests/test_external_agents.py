import os
import unittest
from unittest.mock import Mock, patch

from unstable.external_agents import AzureAIAgent, azure_ai_api_key, azure_ai_endpoint


class AzureAgentTests(unittest.TestCase):
    ENV_NAMES = (
        "AZURE_AI_ENDPOINT", "AZURE_OPENAI_ENDPOINT", "AZURE_FOUNDRY_ENDPOINT",
        "AZURE_AI_RESOURCE", "AZURE_OPENAI_RESOURCE", "ANTHROPIC_FOUNDRY_RESOURCE",
        "AZURE_AI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_INFERENCE_CREDENTIAL",
        "ANTHROPIC_FOUNDRY_API_KEY",
    )

    def setUp(self):
        self.saved_env = {name: os.environ.get(name) for name in self.ENV_NAMES}
        for name in self.ENV_NAMES:
            os.environ.pop(name, None)

    def tearDown(self):
        for name in self.ENV_NAMES:
            os.environ.pop(name, None)
        for name, value in self.saved_env.items():
            if value is not None:
                os.environ[name] = value

    def test_azure_endpoint_from_saved_foundry_resource(self):
        os.environ["ANTHROPIC_FOUNDRY_RESOURCE"] = "research-resource"
        self.assertEqual(
            azure_ai_endpoint(),
            "https://research-resource.services.ai.azure.com/openai/v1/",
        )

    def test_azure_endpoint_preserves_explicit_api_path(self):
        os.environ["AZURE_AI_ENDPOINT"] = "https://example.services.ai.azure.com/models"
        self.assertEqual(azure_ai_endpoint(), "https://example.services.ai.azure.com/models/")

    def test_azure_key_uses_saved_foundry_key(self):
        os.environ["ANTHROPIC_FOUNDRY_API_KEY"] = "secret"
        self.assertEqual(azure_ai_api_key(), "secret")

    def test_azure_agent_calls_deployment(self):
        os.environ["AZURE_AI_ENDPOINT"] = "https://example.services.ai.azure.com/openai/v1"
        os.environ["AZURE_AI_API_KEY"] = "secret"
        completion = Mock()
        completion.choices = [Mock(message=Mock(content="  [Cooperate]  "))]
        client = Mock()
        client.chat.completions.create.return_value = completion

        with patch("openai.OpenAI", return_value=client) as openai_client:
            agent = AzureAIAgent("DeepSeek-V4-flash", max_tokens=64, temperature=0.0)
            self.assertEqual(agent("choose"), "[Cooperate]")

        openai_client.assert_called_once_with(
            base_url="https://example.services.ai.azure.com/openai/v1/",
            api_key="secret",
            default_headers={"api-key": "secret"},
        )
        client.chat.completions.create.assert_called_once_with(
            model="DeepSeek-V4-flash",
            messages=[
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": "choose"},
            ],
            max_tokens=64,
            temperature=0.0,
        )


if __name__ == "__main__":
    unittest.main()
