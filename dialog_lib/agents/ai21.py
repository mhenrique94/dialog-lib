import os
from .abstract import AbstractDialog, AbstractLCEL
from langchain_ai21 import AI21Embeddings, ChatAI21


class DialogOpenAI(AbstractDialog):
    def __init__(self, *args, **kwargs):
        model = kwargs.pop("model", "jamba-instruct-preview")
        temperature = kwargs.pop("temperature", 0.1)
        kwargs["model_class"] = ChatAI21(
            model=model,
            temperature=temperature,
            api_key=kwargs.get("llm_api_key"),
        )
        super().__init__(*args, **kwargs)

    def postprocess(self, output):
        return output.get("text")


class DialogLCELOpenAI(AbstractLCEL):
    def __init__(self, *args, **kwargs):
        self.api_key = kwargs.get("llm_api_key") or os.environ.get("AI21_API_KEY")
        kwargs["model_class"] = ChatAI21(
            model=kwargs.pop("model"),
            temperature=kwargs.pop("temperature"),
            api_key=self.api_key,
        )
        kwargs["embedding_llm"] = AI21Embeddings(api_key=self.api_key)
        super().__init__(*args, **kwargs)
