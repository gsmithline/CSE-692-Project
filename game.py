import torch
from dataclasses import dataclass
from typing import List, Dict, Optional
from utils.offer import Offer

@dataclass
class GameHistory:
    """Stores the complete history and state of a negotiation game"""
    
    # Agent names/identifiers
    agent_1_name: str
    agent_2_name: str
    
    # Game state
    num_items: int
    items: torch.Tensor  # Vector of available quantities for each item
    
    # Agent preferences
    agent_1_values: torch.Tensor  # Value vector for agent 1
    agent_2_values: torch.Tensor  # Value vector for agent 2
    
    # Outside values
    agent_1_outside_value: float  # Outside value for agent 1
    agent_2_outside_value: float  # Outside value for agent 2
    
    # Negotiation history
    agent_1_offers: List[Offer] = []  # List of offers made by agent 1
    agent_2_offers: List[Offer] = []  # List of offers made by agent 2
    
    def __post_init__(self):
        self.num_items = len(self.items)
    
    def add_offer(self, agent_idx: int, offer: Offer):
        """Records an offer made by one of the agents"""
        if agent_idx == 0:
            self.agent_1_offers.append(offer)
        else:
            self.agent_2_offers.append(offer)
    
    def get_offers(self, agent_idx: int) -> List[Offer]:
        """Gets all offers made by specified agent"""
        return self.agent_1_offers if agent_idx == 0 else self.agent_2_offers
