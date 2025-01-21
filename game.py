import torch
from dataclasses import dataclass
from typing import List, Dict, Optional
from utils.offer import Offer
import itertools

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
    
    def compute_pareto_frontier(self):
        """Computes the Pareto frontier for the game"""
        import itertools
        
        # Generate all possible allocations of nondivisible items
        # For item i, we can allocate from 0..items[i] units to agent 1
        all_allocations = itertools.product(
            *[range(int(self.items[i].item()) + 1) for i in range(self.num_items)]
        )
        # Compute agent values for each allocation
        allocations_with_values = []
        for allocation in all_allocations:
            agent1_value = 0.0
            agent2_value = 0.0
            for i, amount_for_agent1 in enumerate(allocation):
                amount_for_agent2 = int(self.items[i].item()) - amount_for_agent1
                agent1_value += amount_for_agent1 * float(self.agent_1_values[i].item())
                agent2_value += amount_for_agent2 * float(self.agent_2_values[i].item())
            allocations_with_values.append((allocation, agent1_value, agent2_value))
        
        # Determine which allocations lie on the Pareto frontier
        frontier_allocations = []
        for i, (alloc_i, val1_i, val2_i) in enumerate(allocations_with_values):
            dominated = False
            for j, (alloc_j, val1_j, val2_j) in enumerate(allocations_with_values):
                if j != i:
                    # Allocation j dominates allocation i if
                    # val1_j >= val1_i and val2_j >= val2_i, with at least one strictly greater
                    if (val1_j >= val1_i and val2_j >= val2_i) and (val1_j > val1_i or val2_j > val2_i):
                        dominated = True
                        break
            if not dominated:
                frontier_allocations.append(alloc_i)
        
        # Format the Pareto frontier as a list of dicts
        result = []
        for alloc in frontier_allocations:
            agent1_split = list(alloc)
            agent2_split = [
                int(self.items[i].item()) - agent1_split[i] for i in range(self.num_items)
            ]
            result.append(
                {
                    "agent1": agent1_split,
                    "agent2": agent2_split
                }
            )
        
        return result
