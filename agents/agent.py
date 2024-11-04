from abc import ABC, abstractmethod
from utils.offer import Offer


class Agent(ABC):
    @abstractmethod
    def give_offer(self, prompt: str) -> Offer | bool:
        """
        Given a prompt describing the current game state, return either:
        - An Offer object containing the player number and proposed item allocation
        - True to accept the current offer
        - False to walk away
        """
        pass
