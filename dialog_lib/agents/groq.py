from .abstract import AbstractDialog
from langchain_groq import ChatGroq


class DialogGroq(AbstractDialog):
    def __init__(self, *args, **kwargs):
        model = kwargs.pop("model", "llama3-8b-8192")
        temperature = kwargs.pop("temperature", 0.1)
        kwargs["model_class"] = ChatGroq(
            model=model,
            temperature=temperature,
            api_key=kwargs.get("llm_api_key"),
        )
        super().__init__(*args, **kwargs)

    def postprocess(self, output):
        return output.get("text")
