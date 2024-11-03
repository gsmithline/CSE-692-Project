from abc import ABC, abstractmethod
from offer import Offer
import openai
from llamaapi import LlamaAPI

class Agent(ABC):
    def __init__(self, llm_type="llama", api_key=None):
        if llm_type == "llama":
            self.llm = LlamaAPI(api_key or "LA-2126f1176b7a452b9f183d94c9fcaa44129183146a41482d819595b9dc9b6c6f")
        elif llm_type == "openai":
            self.llm = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Invalid LLM type: {llm_type}") 
    @abstractmethod
    def give_offer(self, prompt: str) -> Offer | bool:
        """
        Given a prompt describing the current game state, return either:
        - An Offer object containing the player number and proposed item allocation
        - True to accept the current offer
        - False to walk away
        """
        pass
