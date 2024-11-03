from offer import Offer
from make_prompt import make_prompt
import numpy as np
import json


class NegotitaionGame:
    def __init__(self, player1_agent, player2_agent, num_items=4, item_value_range=[0, 101], gamma=0.9, max_rounds=10, outside_offer_value_range=[0, 101]):
        if type(num_items) == int:
            self.items = np.random.poisson(4, num_items)
            self.num_items = num_items
        else:
            self.items = num_items
            self.num_items = len(num_items)

        self.players = [player1_agent, player2_agent]
        
        self.item_values = item_value_range
        self.gamma = gamma
        self.max_rounds = max_rounds
        self.outside_offer_value_range = outside_offer_value_range  # Store the range
        self.outside_offer_values = None  # Will store the actual values
        self.player_values = {0: None, 1: None}
        self.reset()

    def reset(self):
        self.player_values[0] = np.random.randint(self.item_values[0], self.item_values[1], self.num_items)
        self.player_values[1] = np.random.randint(self.item_values[0], self.item_values[1], self.num_items)
        # Fix: Use outside_offer_value_range instead of outside_offer_values
        self.outside_offer_values = np.random.randint(
            self.outside_offer_value_range[0], 
            self.outside_offer_value_range[1], 
            2
        )
        self.current_player = 0
        self.history = {0: [], 1: []}
        self.current_offer = None
        self.in_progress = True

    def step(self):  # Remove the 'offer' parameter
        """Execute one step of the negotiation"""
        agent = self.players[self.current_player]
    
        prompt = make_prompt(
            T=self.num_items,
            quantities=self.items,
            V=self.item_values[1],
            values=self.player_values[self.current_player],
            W1=self.outside_offer_value_range[1],
            W2=self.outside_offer_value_range[0],
            w=self.outside_offer_values[self.current_player],
            R=self.max_rounds,
            g=self.gamma,
            r = (len(self.history[0]) + len(self.history[1])) // 2 + 1,
            history=self.history,
            current_offer=self.current_offer,
            player_num=self.current_player  # Add this line
        ) 
        print(prompt)

        # Get agent's response
        offer = agent.give_offer(prompt)

        if offer is True:  # Accept current offer
            self.in_progress = False
        elif offer is False:  # Walk away
            self.in_progress = False
            self.current_offer = None
        else:  # New offer
            self.current_offer = offer
            if self.current_player == 0:
                self.history[0].append(offer)
            else:
                self.history[1].append(offer)

        self.current_player = 1 - self.current_player

    def run(self):
        """Run the game until completion or max rounds reached"""
        while self.in_progress and len(self.history[0]) + len(self.history[1]) < self.max_rounds:
            self.step()
            
            print(f"Round {len(self.history[0]) + len(self.history[1])}")
            print(f"Current player: {self.current_player + 1}")
            print(f"Current offer: {self.current_offer}")
        
        print("\nGame Complete!")
        if self.current_offer:
            print("Deal reached!")
        else:
            print("No deal reached.")

class GameEvaluator:
    def __init__(self, game: NegotitaionGame):
        self.game = game

    def evaluate_outside_offer_consistency(self):
        for player in [0, 1]:
            for offer in self.game.history[player]:
                if isinstance(offer, Offer):
                    given_value = np.dot(self.game.player_values[player], offer.offer)
                    
                    total_value = np.dot(self.game.player_values[player], self.game.items)
                    kept_value = total_value - given_value
                    
                    if kept_value < self.game.outside_offer_values[player]:
                        return False
                        
        return True
