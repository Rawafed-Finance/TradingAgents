from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def chat(self, prompt, **kwargs):
        pass

class OpenAILLM(BaseLLM):
    def __init__(self, model="gpt-4", **kwargs):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(model=model, **kwargs)

    def chat(self, prompt, **kwargs):
        # Adapt this if your code uses a different interface
        return self.llm.invoke(prompt, **kwargs)

class LocalLLM(BaseLLM):
    def __init__(self, model_path, **kwargs):
        from llama_cpp import Llama
        if 'n_ctx' not in kwargs:
            kwargs['n_ctx'] = 8192
        if 'n_gpu_layers' not in kwargs:
            kwargs['n_gpu_layers'] = 100
        self.llm = Llama(model_path=model_path, **kwargs)

    def chat(self, prompt, **kwargs):
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 8192
        response = self.llm(prompt, **kwargs)
        return response['choices'][0]['text'] 