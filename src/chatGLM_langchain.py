from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from transformers import AutoModel, AutoTokenizer

# This is minimal customer LLM wrapper based on ChatGLM model
class ChatGLM(LLM):
    tokenizer = AutoTokenizer.from_pretrained("./chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("./chatglm-6b-int4",trust_remote_code=True).float()
    
    model.eval()
        
    @property
    def _llm_type(self) -> str:
        return "ChatGLM"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")


        return self.model.chat(self.tokenizer, prompt, history=[])[0]
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": None}


