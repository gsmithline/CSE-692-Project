#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
from dataclasses import dataclass, field
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
    agent_1_offers: List[Offer] = field(default_factory=list)  # List of offers made by agent 1
    agent_2_offers: List[Offer] = field(default_factory=list)  # List of offers made by agent 2
    
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


# In[2]:


import torch
import random

# Create items tensor
items = torch.tensor([3, 2, 1])

# Create random value tensors between 0 and 1
agent_1_values = torch.tensor([random.random() for _ in range(3)])
agent_2_values = torch.tensor([random.random() for _ in range(3)])

# Create game history
game = GameHistory(
    agent_1_name="Agent 1",
    agent_2_name="Agent 2", 
    num_items=len(items),  # Add the num_items argument
    items=items,
    agent_1_values=agent_1_values,
    agent_2_values=agent_2_values,
    agent_1_outside_value=2.0,
    agent_2_outside_value=2.0
)

# Add the specified offers
offers = [
    torch.tensor([0, 1, 1]),
    torch.tensor([1, 1, 1]), 
    torch.tensor([2, 1, 0])
]

for i, offer in enumerate(offers):
    game.add_offer(0, Offer(0, offer))


# In[3]:


import numpy as np
from utils.offer import Offer

class GameEvaluator:
    def __init__(self, game: GameHistory):
        self.game = game

    def evaluate_outside_offer_consistency(self):
        for player in [0, 1]:
            player_offers = self.game.get_offers(player)
            player_values = self.game.agent_1_values if player == 0 else self.game.agent_2_values
            outside_value = self.game.agent_1_outside_value if player == 0 else self.game.agent_2_outside_value
            
            for offer in player_offers:
                given_value = torch.dot(player_values, offer.offer)
                total_value = torch.dot(player_values, self.game.items)
                kept_value = total_value - given_value
                
                if kept_value < outside_value:
                    return False
                    
        return True
    
    def evaluate_offer_increasing(self):
        for player in [0, 1]:
            opponent = 1 - player
            player_offers = self.game.get_offers(player)
            opponent_offers = self.game.get_offers(opponent)
            player_values = self.game.agent_1_values if player == 0 else self.game.agent_2_values
            
            for i, offer in enumerate(player_offers):
                given_value = torch.dot(player_values, offer.offer)
                total_value = torch.dot(player_values, self.game.items)
                kept_value = total_value - given_value

                if i > 0:
                    prev_offer = opponent_offers[i-1]
                    opp_offer_value = torch.dot(player_values, prev_offer.offer)
                    
                    if kept_value < opp_offer_value:
                        return False
                    
        return True

    def evaluate_envy_free(self, exclude_one_item=False):
        for player in [0, 1]:
            opponent = 1 - player
            player_offers = self.game.get_offers(player)
            opponent_offers = self.game.get_offers(opponent)
            player_values = self.game.agent_1_values if player == 0 else self.game.agent_2_values
            
            for i, offer in enumerate(player_offers):
                if i < len(opponent_offers):
                    opp_offer = opponent_offers[i]
                    player_bundle_value = torch.dot(player_values, self.game.items - offer.offer)
                    opponent_bundle_value = torch.dot(player_values, opp_offer.offer)
                    
                    if exclude_one_item:
                        for j in range(len(self.game.items)):
                            temp_opp_offer = opp_offer.offer.clone()
                            if temp_opp_offer[j] > 0:
                                temp_opp_offer[j] -= 1
                                temp_opponent_bundle_value = torch.dot(player_values, temp_opp_offer)
                                if player_bundle_value >= temp_opponent_bundle_value:
                                    break
                        else:
                            return False
                    else:
                        if player_bundle_value < opponent_bundle_value:
                            return False
        
        return True
    


# In[4]:


# Create sample game history
sample_history = GameHistory(
    agent_1_name="Agent1",
    agent_2_name="Agent2",
    num_items=4,
    items=torch.tensor([3, 2, 4, 1]),
    agent_1_values=torch.tensor([10, 20, 30, 40]),
    agent_2_values=torch.tensor([40, 30, 20, 10]), 
    agent_1_outside_value=50.0,
    agent_2_outside_value=40.0
)

# Add some sample offers
offer1 = Offer(0, torch.tensor([1, 1, 2, 0]))
offer2 = Offer(1, torch.tensor([2, 1, 2, 1])) 
offer3 = Offer(0, torch.tensor([2, 1, 3, 0]))

sample_history.add_offer(0, offer1)
sample_history.add_offer(1, offer2)
sample_history.add_offer(0, offer3)

# Create evaluator and test
evaluator = GameEvaluator(sample_history)

print("Outside offer consistency:", evaluator.evaluate_outside_offer_consistency())
print("Offer increasing:", evaluator.evaluate_offer_increasing())
print("Envy-free (strict):", evaluator.evaluate_envy_free())
print("Envy-free (excluding one item):", evaluator.evaluate_envy_free(exclude_one_item=True))

# Print the value of outside offer and kept items for each offer
for player in [0, 1]:
    player_offers = sample_history.get_offers(player)
    player_values = sample_history.agent_1_values if player == 0 else sample_history.agent_2_values
    outside_value = sample_history.agent_1_outside_value if player == 0 else sample_history.agent_2_outside_value
    
    print(f"\nPlayer {player + 1}:")
    print(f"Outside offer value: {outside_value}")
    
    for i, offer in enumerate(player_offers):
        given_value = torch.dot(player_values, offer.offer)
        total_value = torch.dot(player_values, sample_history.items)
        kept_value = total_value - given_value
        
        print(f"Offer {i + 1}:")
        print(f"  Offer: {offer.offer}")
        print(f"  Player values: {player_values}")
        print(f"  Value calculation:")
        print(f"    Total value: {total_value.item()} = {player_values} · {sample_history.items}")
        print(f"    Given value: {given_value.item()} = {player_values} · {offer.offer}")
        print(f"    Kept value: {kept_value.item()} = {total_value.item()} - {given_value.item()}")
        print(f"  Value of kept items: {kept_value.item()}")

# Print envy-free evaluation details
print("\nEnvy-free evaluation details:")
for player in [0, 1]:
    opponent = 1 - player
    player_offers = sample_history.get_offers(player)
    opponent_offers = sample_history.get_offers(opponent)
    player_values = sample_history.agent_1_values if player == 0 else sample_history.agent_2_values
    
    print(f"\nPlayer {player + 1}:")
    for i, offer in enumerate(player_offers):
        if i < len(opponent_offers):
            opp_offer = opponent_offers[i]
            player_bundle_value = torch.dot(player_values, sample_history.items - offer.offer)
            opponent_bundle_value = torch.dot(player_values, sample_history.items - opp_offer.offer)
            
            print(f"Round {i + 1}:")
            print(f"  Player's offer: {offer.offer}")
            print(f"  Opponent's offer: {opp_offer.offer}")
            print(f"  Player's bundle value: {player_bundle_value.item()} = {player_values} · ({sample_history.items} - {offer.offer})")
            print(f"  Opponent's bundle value (to player): {opponent_bundle_value.item()} = {player_values} · ({sample_history.items} - {opp_offer.offer})")
            
            if player_bundle_value >= opponent_bundle_value:
                print("  Envy-free: Yes")
            else:
                print("  Envy-free: No")
                print("  Checking if removing one item makes it envy-free:")
                for j in range(len(sample_history.items)):
                    temp_opp_offer = opp_offer.offer.clone()
                    if temp_opp_offer[j] > 0:
                        temp_opp_offer[j] -= 1
                        temp_opponent_bundle_value = torch.dot(player_values, sample_history.items - temp_opp_offer)
                        print(f"    Removing item {j + 1}: {temp_opponent_bundle_value.item()} = {player_values} · ({sample_history.items} - {temp_opp_offer})")
                        if player_bundle_value >= temp_opponent_bundle_value:
                            print(f"    Envy-free after removing item {j + 1}: Yes")
                            break
                else:
                    print("    Envy-free after removing any single item: No")


# In[ ]:




