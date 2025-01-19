import numpy as np
from utils.offer import Offer

'''
CIRCLE OF ERRORS
- Addition add description of errors & explain that errors are obstructive to the objective
'''

def make_prompt_circle_3(T: int, quantities: list[int], V: int, values: list[float], W1: int, W2: int, w: int, R: int, g: float, r: int, history: dict, current_offer: Offer = None, player_num: int = 0, p1_outside_offer: list[int] = None, p2_outside_offer: list[int] = None, circle: int = 0, other_player_num: int = 0, my_player_num: int = 0) -> str:
    return f"""
    You and another agent have to negotiate a division of items between the two of you.
    You are Player {my_player_num} and the other agent is Player {other_player_num}.
    There are {T} types of items, called item 1 through item {T}.
    There are {', '.join([f'{q} units of item {i+1}' for i, q in enumerate(quantities)])} to divide.
    Both you and Player {other_player_num} have a private value per unit of each item type.
    These values are drawn from a uniform random distribution, ranging from 1 to {V}.
    Your private values are {', '.join([str(v) + ' for item ' + str(i+1) for i, v in enumerate(values)])}.
    You have a private outside offer drawn from a uniform random distribution ranging from {p1_outside_offer[0] if my_player_num == 1 else p2_outside_offer[0]} to your total value of all items, which is{p1_outside_offer[1] if my_player_num == 1 else p2_outside_offer[1]}. Player {other_player_num} has a private outside offer drawn from a uniform random distribution ranging from 1 to to their total value of all items.
    Your outside offer value is {w}. Your objective is to maximize your value of the outcome of the negotiation game. Remember, you have a guaranteed alternative: your outside offer.

    Before making any counteroffer, you should calculate its total value to you and compare it to your outside offer value of {w}. 
    For example, if you were considering offering the other player 2 units of each item (keeping 3 units of each for yourself), you would calculate:
    3 units of item 1 = 3 × {values[0]} = {3*values[0]} (multiplying units by your value per unit)
    3 units of item 2 = 3 × {values[1]} = {3*values[1]} (multiplying units by your value per unit)
    3 units of item 3 = 3 × {values[2]} = {3*values[2]} (multiplying units by your value per unit)
    3 units of item 4 = 3 × {values[3]} = {3*values[3]} (multiplying units by your value per unit)
    3 units of item 5 = 3 × {values[4]} = {3*values[4]} (multiplying units by your value per unit)
    Total value = {sum([3*values[i] for i in range(T)])} (sum of all item values)
    
    The negotiation proceeds in {R} rounds.
    There is a discount rate gamma = {g}, such that if the process concludes after r rounds the overall value of the negotiation to each player is their value for the outcome multiplied by gamma to the power (r-1).
    At each round, Player 1 takes an action, followed by Player 2.
    The possible actions are to ACCEPT the other player's current offer (if any), make a COUNTEROFFER, or WALK away.  If the game gets to the last round, and player 2 chooses to make a counteroffer, this is treated as a WALK.
    If a player chooses ACCEPT, the negotiation ends in a deal to divide the items according to the accepted offer.
    The value of an outcome is determined by each player's private values per unit of each item and the quantities they receive in the deal. This value is adjusted by the discount factor, which is used to compute the present value of the negotiation outcome.
    If a player chooses WALK, the negotiation ends without a deal, and each player receives the value of their private outside offer.

    In the bargaining game, there are 5 actions that you can execute that would be inconsistent with your objective. These represent undesirable negotiation behaviors that can undermine your payoff or cause you to miss out on better deals. Avoiding these mistakes is crucial to achieving the most advantageous outcome.
    These mistakes are:
    - Mistake 1: Making an offer worse than your previous offer. This occurs when you reject an offer better for you than the one you subsequently propose. 
    - Mistake 2: Making an offer worse for you than your outside offer. This happens if you propose giving away so much that what you keep is worth less than your guaranteed alternative, which is your outside offer.
    - Mistake 3: Offering no items or all items. Offering nothing (or everything) to the opponent (in the early or middle rounds) can be a clear suboptimal move. 
    - Mistake 4: Accepting an offer worse for you than your outside offer. This occurs if you accept a division that yields a payoff lower than your guaranteed fallback.
    - Mistake 5: Walking away from an offer better than your outside offer. This occurs when you reject a division that actually yields a higher payoff than your fallback.

    Please show your reasoning step by step, then provide your action in one of these formats in your response (if you do not do this your response will be invalid):
    {{"action": "ACCEPT"}} - to accept the current offer
    {{"action": "WALK"}} - to walk away from negotiations  
    {{"action": "COUNTEROFFER", "offer": [n1, n2, ...]}} - where n1, n2, ... are numbers representing your counteroffer

    Any response not in these exact formats will be invalid and treated as a WALK. If you provide a counteroffer, it must be a valid offer, otherwise it will be treated as a WALK.

    It is now round {r}.
    """