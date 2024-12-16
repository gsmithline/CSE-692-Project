from abc import ABC, abstractmethod
from utils.offer import Offer
import openai
from llamaapi import LlamaAPI
import os
class Agent(ABC):
    def __init__(self, llm_type="llama", api_key=None):
        self.llm_type = llm_type
        self.api_key = api_key
        pass
    
    @abstractmethod
    def give_offer(self, prompt: str) -> Offer | bool:
        """
        Given a prompt describing the current game state, return either:
        - An Offer object containing the player number and proposed item allocation
        - True to accept the current offer
        - False to walk away
        """
        pass
