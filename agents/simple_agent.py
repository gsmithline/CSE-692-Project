from agents.agent import Agent
from offer import Offer
import numpy as np

class SimpleAgent(Agent):
    def __init__(self):
        super().__init__()  # No LLM needed for this agent
        
    def give_offer(self, prompt: str) -> Offer | bool:
        # Simple strategy: Always offer to take 60% of each item
        quantities = [int(q) for q in prompt.split("units of item")[1:]]
        my_offer = [int(0.6 * q) for q in quantities]
        return Offer(player=self.player_num, offer=my_offer)