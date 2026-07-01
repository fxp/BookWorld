from .BaseLLM import BaseLLM
from openai import OpenAI
import os

# BigModel / Zhipu AI (智谱) exposes an OpenAI-compatible endpoint.
# Docs: https://open.bigmodel.cn/dev/api  ->  base_url ends in /api/paas/v4/
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


class GLM(BaseLLM):
    """Adapter for the GLM family (glm-4-plus, glm-4-air, glm-4-flash, charglm-3, ...)
    served by BigModel / Zhipu AI through its OpenAI-compatible API."""

    def __init__(self, model="glm-4-plus"):
        super(GLM, self).__init__()
        api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("BIGMODEL_API_KEY")
        self.client = OpenAI(
            api_key=api_key,
            base_url=ZHIPU_BASE_URL,
        )
        self.model_name = model
        self.messages = []

    def initialize_message(self):
        self.messages = []

    def ai_message(self, payload):
        self.messages.append({"role": "assistant", "content": payload})

    def system_message(self, payload):
        self.messages.append({"role": "system", "content": payload})

    def user_message(self, payload):
        self.messages.append({"role": "user", "content": payload})

    def get_response(self, temperature=0.8):
        kwargs = dict(
            model=self.model_name,
            messages=self.messages,
            temperature=temperature,
            stream=False,
        )
        # glm-5 / glm-4.5 are hybrid reasoning models: with thinking ON they spend the
        # whole token budget reasoning and return empty content. Disable for direct answers.
        import re
        if re.search(r"glm-(5|4\.5)", self.model_name, re.I):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def chat(self, text):
        self.initialize_message()
        self.user_message(text)
        return self.get_response()

    def print_prompt(self):
        for message in self.messages:
            print(message)
