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
    
   
    def offer_no_items_or_all_items(self, game_round = 0, player = 0):
        offer = self.game.history[player][game_round]
        if isinstance(offer, Offer):
            all_zeros = all(val == 0 for val in offer.offer)
            all_max = all(val == item for val, item in zip(offer.offer, self.game.items))
            return all_zeros or all_max
        return False
    '''
    def check_pathologies(self, round_num, game_num, player, is_final_action = False):
        game_num = game_num 
        round_num = round_num 
        print(f"Game {game_num} Round {round_num} Player {player + 1}")
        previous_offer, your_side_of_previous_offer = None, None
        game = self.game
        if not is_final_action and round_num >= len(game.history[player]):
            return None 
        if is_final_action:
            # Use the last offer from the other player
            other_player = 1 - player
            offer = game.history[other_player][-1]
        else:
            offer = game.history[player][round_num]
        value = game.player_values[player]
        items = game.items
        your_side_of_current_offer = items - offer.offer
        outside_offer = game.outside_offer_values[player]
        if round_num > 0:
            previous_offer = game.history[player][round_num - 1]
            your_side_of_previous_offer = items - previous_offer.offer

        #Outside offer consistency
        #check if offer is worse than outside offer 
        #check if accepted an offer worse than outside offer
        #check if made offer that is worse for you than your outside offer
        #check if walked away from an offer that is better than your outside offer
        #check if offer no items or all items (maybe not a pathology)
        #outside_offer_consistency = (sum(your_side_of_current_offer * value) * game.gamma ** (round_num)) >= outside_offer
        accepting_an_offer_worse_than_outside_offer = (np.dot(your_side_of_current_offer, value) * game.gamma ** (round_num)) < outside_offer and (game.players[player].result == True) 
        making_an_offer_worse_for_you_than_your_outside_offer = (np.dot(your_side_of_current_offer, value) * game.gamma ** (round_num) < outside_offer and isinstance(offer, Offer) and not isinstance(game.players[player].result, bool))
        walking_away_from_an_offer_better_than_your_outside_offer = (np.dot(your_side_of_current_offer, value) * game.gamma ** (round_num)) > outside_offer and (game.players[player].result == False) 
        offer_no_items_or_all_items = self.offer_no_items_or_all_items(round_num, player)
    


        return {
            "game_num": game_num +1,
            "round_num": round_num + 1,
            "player": player + 1,
            "discount_rate": game.gamma ** (round_num),
            "offer": list(offer.offer) if isinstance(offer.offer, str) else offer.offer,
            "value": value,
            "previous_offer": list(previous_offer.offer) if previous_offer and isinstance(previous_offer.offer, str) else (previous_offer.offer if previous_offer else None),
            "items": items,
            "your_side_of_current_offer": your_side_of_current_offer,
            "outside_offer": outside_offer,
            #"outside_offer_consistency": outside_offer_consistency,
            "accepting_an_offer_worse_than_outside_offer": accepting_an_offer_worse_than_outside_offer,
            "making_an_offer_worse_for_you_than_your_outside_offer": making_an_offer_worse_for_you_than_your_outside_offer,
            "walking_away_from_an_offer_better_than_your_outside_offer": walking_away_from_an_offer_better_than_your_outside_offer,
            "offer_no_items_or_all_items": offer_no_items_or_all_items
        }
        '''
    def check_pathologies(self, round_num, game_num, player, is_final_action=False):
        game = self.game
        if not is_final_action and round_num >= len(game.history[player]):
            return None

        game = self.game
        value = game.player_values[player]
        items = game.items
        outside_offer = game.outside_offer_values[player]
        
        # Get current offer and calculate values
        if is_final_action:
            other_player = 1 - player
            offer = game.history[other_player][-1]
        else:
            offer = game.history[player][round_num]
        
        your_side_of_current_offer = items - np.array(offer.offer)
        current_value = np.dot(your_side_of_current_offer, value) * game.gamma ** round_num
        outside_offer_value = outside_offer * game.gamma ** round_num
        making_offer_worse_than_previous = False
        if round_num > 0 and len(game.history[1-player]) >= round_num:
            previous_offer = game.history[1-player][round_num-1]
            your_side_of_previous_offer = items - np.array(previous_offer.offer)
            previous_value = np.dot(your_side_of_previous_offer, value) * game.gamma ** (round_num - 1)
            making_offer_worse_than_previous = (current_value < previous_value and 
                                            isinstance(offer, Offer) and 
                                            not isinstance(game.players[player].result, bool))
        # Check pathologies
        accepting_worse = (current_value < outside_offer_value and 
                  game.players[player].result is True and
                  isinstance(game.players[player].action, str) and
                  game.players[player].action == "ACCEPT")
        
        making_worse_offer = (current_value < outside_offer_value and 
                            isinstance(offer, Offer) and 
                            not isinstance(game.players[player].result, bool))
        
        walking_from_better = (current_value > outside_offer_value and 
                            game.players[player].result is False and 
                            isinstance(game.players[player].action, str) and
                            game.players[player].action == "WALK")
        
        offer_extremes = (all(q == 0 for q in offer.offer) or 
                        all(q == item for q, item in zip(offer.offer, items)))

        result = {
            "game_num": game_num + 1,
            "round_num": round_num + 1 ,
            "player": player + 1,
            "discount_rate": game.gamma ** round_num,
            "offer": list(offer.offer) if isinstance(offer.offer, str) else offer.offer,
            "value": value,
            "items": items,
            "your_side_of_current_offer": your_side_of_current_offer,
            "outside_offer": outside_offer,
            "accepting_an_offer_worse_than_outside_offer": accepting_worse,
            "making_an_offer_worse_for_you_than_your_outside_offer": making_worse_offer,
            "walking_away_from_an_offer_better_than_your_outside_offer": walking_from_better,
            "offer_no_items_or_all_items": offer_extremes,
            "making_offer_worse_than_previous": making_offer_worse_than_previous
        }
        print(result)
        return result
    
    def check_final_action(self, game_num, player, final_round):
        game_num = game_num + 1

        
        game = self.game
        value = game.player_values[player.player_num]
        items = game.items
        outside_offer = game.outside_offer_values[player.player_num]
        
        # Get the last offer that was on the table
        other_player = 1 - player.player_num
        print(game.history[other_player])   
        if len(game.history[other_player]) > 0:
            last_offer_to_you = game.history[other_player][-1] #this player is not offering only accepting or walking away
            value_of_last_offer_to_you = np.dot(value, last_offer_to_you.offer) * game.gamma ** (final_round)
            
            # Initialize pathology flags
            walking_away_from_better = False
            accepting_worse = False
            outside_offer_value = outside_offer * game.gamma ** (final_round)
            
            # Check pathologies based on final action
            if game.final_action_player.action == "WALK" or game.final_action_player.action == "COUNTEROFFER":
                walking_away_from_better = (value_of_last_offer_to_you > outside_offer_value)
            elif game.final_action_player.action == "ACCEPT" or game.final_action_player.action == "COUNTEROFFER":
                accepting_worse = (value_of_last_offer_to_you < outside_offer_value)
            result = {
                'game_num': game_num,
                'round_num': final_round + 1,
                'player': player.player_num + 1,
                'discount_rate': game.gamma ** (final_round),
                'offer': None,
                'value': value,
                'previous_offer': last_offer_to_you.offer,
                'items': items,
                'your_side_of_current_offer': None,
                'outside_offer': outside_offer,
                #'outside_offer_consistency': outside_offer_rational,
                'accepting_an_offer_worse_than_outside_offer': accepting_worse, 
                'making_an_offer_worse_for_you_than_your_outside_offer': None, #not applicable
                'walking_away_from_an_offer_better_than_your_outside_offer': walking_away_from_better,
                'offer_no_items_or_all_items': None #not applicable
            }
        print(result)
        return result
        