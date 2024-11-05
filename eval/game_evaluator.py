import numpy as np
from utils.offer import Offer
from game_runner import NegotitaionGame

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
    
    def evaluate_offer_increasing(self):
        for player in [0, 1]:
            opponent = 1 - player
            for i, offer in enumerate(self.game.history[player]):
                if isinstance(offer, Offer):
                    given_value = np.dot(self.game.player_values[player], offer.offer)
                    total_value = np.dot(self.game.player_values[player], self.game.items)
                    kept_value = total_value - given_value

                    # Should this have a discount?
                    if player == 0 and i >= 1:
                        prev_offer = self.game.history[opponent][i]
                        opp_offer_value = np.dot(self.game.player_values[player], prev_offer.offer)
                    else:
                        prev_offer = self.game.history[opponent][i-1]
                        opp_offer_value = np.dot(self.game.player_values[player], prev_offer.offer)
                    
                    if kept_value < opp_offer_value:
                        return False
                        
        return True
