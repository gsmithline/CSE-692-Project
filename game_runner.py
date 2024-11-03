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
        
        self.item_value_range = item_value_range
        self.gamma = gamma
        self.max_rounds = max_rounds
        self.outside_offer_value_range = outside_offer_value_range
        self.player_values = {0: None, 1: None}
        self.reset()

    def reset(self):
        self.player_values[0] = np.random.randint(self.item_values[0], self.item_values[1], self.num_items)
        self.player_values[1] = np.random.randint(self.item_values[0], self.item_values[1], self.num_items)
        self.outside_offer_values = np.random.randint(self.outside_offer_values[0], self.outside_offer_values[1], 2)
        self.current_player = 0
        self.history = {0: [], 1: []}
        self.current_offer = None
        self.in_progress = True

    def step(self, offer: Offer):
        agent = self.players[self.current_player]
        offer = agent.give_offer(make_prompt(
            T=self.max_rounds,
            quantities=self.items,
            V=self.item_value_range[1],
            values=self.player_values[self.current_player],
            W=self.outside_offer_value_range[1],
            w=self.outside_offer_value_range[self.current_player],
            R=self.max_rounds,
            g=self.gamma,
            r=len(self.history[0]) + len(self.history[1])
        ))

        if offer == True:
            self.in_progress = False
        
        if self.current_player == 0:
            self.history[0].append(offer)
        else:
            self.history[1].append(offer)

        self.current_player = 1 - self.current_player


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
