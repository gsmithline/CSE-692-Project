from utils.offer import Offer
from prompts.make_prompt import make_prompt
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
        self.current_round = 0

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

        if offer == True or offer == False:
            self.in_progress = False
        
        if self.current_player == 0:
            self.history[0].append(offer)
        else:
            self.history[1].append(offer)

        self.current_player = 1 - self.current_player
        if self.current_player == 0:
            self.current_round += 1

        if self.current_round >= self.max_rounds:
            self.in_progress = False


