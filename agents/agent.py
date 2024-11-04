from abc import ABC, abstractmethod
from offer import Offer
import openai
from llamaapi import LlamaAPI
import os
class Agent(ABC):
    def __init__(self, llm_type="llama", api_key=None):
        if llm_type == "llama":
            if api_key is None:
                try:
                    # Get the directory of the current file
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    # Go up one level to project root and look for the API key file
                    key_path = os.path.join(os.path.dirname(current_dir), 'LLAMA_API_KEY.txt')
                    with open(key_path, 'r') as f:
                        api_key = f.read().strip()
                except FileNotFoundError:
                    raise ValueError("No API key provided and couldn't find LLAMA_API_KEY.txt")
            self.llm = LlamaAPI(api_key)
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
