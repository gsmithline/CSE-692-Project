from utils.offer import Offer
from prompts.make_prompt import make_prompt
from prompts.make_prompt_bargain import make_prompt_bargain
import numpy as np
import json
import pandas as pd

class NegotitaionGame:
    def __init__(self, player1_agent, player2_agent, num_items=4, item_value_range=[1, 101], gamma=0.9, max_rounds=10, game_results=pd.DataFrame(), envy_results=pd.DataFrame(), circle1: int = 0, circle2: int = 0):
        if type(num_items) == int:
            self.items = np.random.poisson(4, num_items)
            self.num_items = num_items
        else:
            self.items = num_items
            self.num_items = len(num_items)

        self.players = [player1_agent, player2_agent]
        
        self.item_value_range = item_value_range
        self.game_results = game_results
        self.envy_results = envy_results
        self.gamma = gamma
        self.max_rounds = max_rounds
        #self.outside_offer_value_range = outside_offer_value_range  
        self.outside_offer_values = None  
        self.player_values = {0: None, 1: None}
        self.reset()
        self.final_action_player = self.players[1] #default to player 2 as the final action player of final player in final round
        self.circle1 = circle1
        self.circle2 = circle2
        self.valid_walk = None
        self.current_prompt = None



    def reset(self):
        self.player_values[0] = np.random.randint(self.item_value_range[0], self.item_value_range[1], self.num_items) 
        self.player_values[1] = np.random.randint(self.item_value_range[0], self.item_value_range[1], self.num_items)
        #self.player_values[0] = np.clip(np.round(np.random.normal(50, 10, self.num_items)), self.item_value_range[0], self.item_value_range[1]).astype(int)
        #self.player_values[1] = np.clip(np.round(np.random.normal(50, 10, self.num_items)), self.item_value_range[0], self.item_value_range[1]).astype(int)

        

        total_value_player0 = int(np.ceil(np.dot(self.items, self.player_values[0]) * 1)) #NOTE: CHANGE % FOR EXPERIMENTS 
        total_value_player1 = int(np.ceil(np.dot(self.items, self.player_values[1]) * 1))
        
        
        self.outside_offer_values = np.array([
            np.random.randint(1, total_value_player0),
            np.random.randint(1, total_value_player1)
        ])
        
        self.current_player = 0
        self.history = {0: [], 1: []}
        self.current_offer = None
        self.in_progress = True
        self.current_round = 0
    def step(self, example_offer_less_than_outside_offer_self: list[int] = None):  
        """Execute one step of the negotiation"""
        agent = self.players[self.current_player]
    
        prompt = make_prompt(
            T=self.num_items,
            quantities=self.items,
            V=self.item_value_range[1],
            values=self.player_values[self.current_player],
            W1= int(np.ceil(np.dot(self.items, self.player_values[0]) * 1)) if self.current_player == 0 else int(np.ceil(np.dot(self.items, self.player_values[1]) * 1)), #NOTE: CHANGE % FOR EXPERIMENTS 
            W2=1,
            w=self.outside_offer_values[self.current_player],
            R=self.max_rounds,
            g=self.gamma,
            r = (len(self.history[0]) + len(self.history[1])) // 2 + 1,
            history=self.history,
            current_offer=self.current_offer,
            player_num=self.current_player,
            p1_outside_offer=[1, int(np.ceil(np.dot(self.items, self.player_values[0]) * 1))], #NOTE: CHANGE % FOR EXPERIMENTS 
            p2_outside_offer=[1, int(np.ceil(np.dot(self.items, self.player_values[1]) * 1))],
            circle = self.circle1 if self.current_player == 0 else self.circle2,
            example_offer_less_than_outside_offer_self=example_offer_less_than_outside_offer_self
        )
        agent.current_prompt = prompt
        print(prompt)


        # Get agent's response
        offer = agent.give_offer(prompt)

        evaluator = GameEvaluator(self)
        try:
            if evaluator.validate_offer(offer):
                if offer is True:  # Accept current offer
                    self.in_progress = False
                    self.final_action_player = self.players[self.current_player]
                elif offer is False:  # Walk away
                    self.in_progress = False
                    self.current_offer = None
                    self.final_action_player = self.players[self.current_player]
                elif len(offer.offer) != self.num_items:
                    print("Invalid offer: Incorrect number of items")
                    self.in_progress = False
                    self.current_offer = None
                    self.final_action_player = self.players[self.current_player]
                    self.final_action_player.action = "INVALID WALK"
                else:  # New offer
                    self.current_offer = offer
                    if self.current_player == 0:
                        self.history[0].append(offer)
                    else:
                        self.history[1].append(offer)
        except ValueError as e: #If offer is invalid, terminate game and treat as walk away
            print(f"Game terminated due to invalid offer: {str(e)}")
            self.in_progress = False
            self.current_offer = None
            self.final_action_player = self.players[1 -self.current_player]
            self.final_action_player.action = "INVALID WALK"
            offer = False
            
        '''
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
        '''
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

    def validate_offer(self, offer: Offer):
        """
        Validates if an offer is legal:
        - All quantities must be non-negative
        - No quantity can exceed available items
        
        Returns:
        - True if offer is valid
        - False if offer is invalid
        """
        if type(offer) is bool:
            return True
        if any(q < 0 for q in offer.offer):
            raise ValueError("Invalid offer: Negative quantities are not allowed")
            
        # Check if quantities exceed available items
        if any(q > max_q for q, max_q in zip(offer.offer, self.game.items)):
            raise ValueError("Invalid offer: Quantities exceed available items")
        #check if all quantities are integers
        if any(not isinstance(q, int) for q in offer.offer):
            raise ValueError("Invalid offer: Quantities must be integers")
        #check length is 5
        if len(offer.offer) != 5:
            raise ValueError("Invalid offer: Quantities must of items must be 5")

        return True
