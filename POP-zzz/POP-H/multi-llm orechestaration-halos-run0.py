class LLMRouter:
    def __init__(self):
        self.providers = {
            LLMProvider.OPENAI: OpenAIClient(),
            LLMProvider.ANTHROPIC: AnthropicClient(),
            LLMProvider.GEMINI: GeminiClient(),
            LLMProvider.LOCAL: LocalLLMClient()
        }
        self.task_mapping = {
            "creative": LLMProvider.OPENAI,
            "analysis": LLMProvider.ANTHROPIC,
            "coding": LLMProvider.GEMINI,
            "confidential": LLMProvider.LOCAL
        }

    def get_response(self, prompt: str, task_type: str = "general") -> str:
        provider = self.task_mapping.get(task_type, LLMProvider.OPENAI)
        try:
            return self.providers[provider].generate(prompt)
        except Exception as e:
            print(f"Primary provider failed, trying fallbacks: {e}")
            for provider in self.providers.values():
                try:
                    return provider.generate(prompt)
                except:
                    continue
            raise Exception("All LLM providers failed")