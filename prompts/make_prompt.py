from utils.offer import Offer
import numpy as np
def make_prompt(T: int, quantities: list[int], V: int, values: list[float], W1: int, W2: int, w: int, R: int, g: float, r: int, history: dict, current_offer: Offer = None, player_num: int = 0, p1_outside_offer: list[int] = None, p2_outside_offer: list[int] = None) -> str:
    
    my_player_num = player_num + 1  
    other_player_num = 2 if my_player_num == 1 else 1
    
   
    history_str = ""
    for round_num in range(len(history[0]) + len(history[1])):
        player = round_num % 2
        round_idx = round_num // 2  
        if round_idx < len(history[player]):
            offer = history[player][round_idx]
        if isinstance(offer, Offer):
            
            history_str += f"\nRound {round_idx + 1}: Player {player + 1} offered {offer.offer}"
        elif offer is True:
            history_str += f"\nRound {round_idx + 1}: Player {player + 1} ACCEPTED"
        elif offer is False:
            history_str += f"\nRound {round_idx + 1}: Player {player + 1} WALKED away"  

    current_offer_str = f"\nCurrent offer on the table (the amount of each item being offered to you): {current_offer.offer if current_offer else 'None'}"
    
    
    if r == 1 and my_player_num == 1:
        action_prompt = f"""
        What is your action? As the first player, you can:
        - WALK to walk away
        - A list of numbers [n1, n2, ...] representing your initial offer (what you give to Player 2)"""
    elif current_offer is None:
        action_prompt = f"""
        What is your action? You can:
        - WALK to walk away
        - A list of numbers [n1, n2, ...] representing your offer (what you give to Player {other_player_num})"""
    else:
        action_prompt = f"""
    What is your action? You can:
    - ACCEPT to accept the current offer
    - WALK to walk away
    - A list of numbers [n1, n2, ...] representing your counteroffer (what you give to Player {other_player_num})"""

    return f"""
    You and another agent have to negotiate a division of items between the two of you.
    You are Player {my_player_num} and the other agent is Player {other_player_num}.
    There are {T} types of items, called item 1 through item {T}.
    There are {', '.join([f'{q} units of item {i+1}' for i, q in enumerate(quantities)])} to divide.
    Both you and Player {other_player_num} have a private value per unit of each item type.
    These values are drawn from a uniform random distribution, ranging from 0 to {V}.
    Your private values are {', '.join([str(v) + ' for item ' + str(i+1) for i, v in enumerate(values)])}.
    You have a private outside offer drawn from a uniform random distribution ranging from {p1_outside_offer[0] if my_player_num == 1 else p2_outside_offer[0]} to {p1_outside_offer[1] if my_player_num == 1 else p2_outside_offer[1]}. Player {other_player_num} has a private outside offer drawn from a uniform random distribution ranging from {p2_outside_offer[0] if my_player_num == 1 else p1_outside_offer[0]} to {p2_outside_offer[1] if my_player_num == 1 else p1_outside_offer[1]}.
    Your outside offer value is {w}. Your objective is to maximize your value of the outcome of the negotiation game. Remember, you have a guaranteed alternative: your outside offer.

    Before making any counteroffer, you should calculate its total value to you and compare it to your outside offer value of {w}. For example, if you were considering offering the other player 2 units of each item (keeping 3 units of each for yourself), you would calculate:
    3 units of item 1 = 3 × {values[0]} = {3*values[0]} (multiplying units by your value per unit)
    3 units of item 2 = 3 × {values[1]} = {3*values[1]} (multiplying units by your value per unit)
    3 units of item 3 = 3 × {values[2]} = {3*values[2]} (multiplying units by your value per unit)
    3 units of item 4 = 3 × {values[3]} = {3*values[3]} (multiplying units by your value per unit)
    3 units of item 5 = 3 × {values[4]} = {3*values[4]} (multiplying units by your value per unit)
    Total value = {sum([3*values[i] for i in range(T)])} (sum of all item values)
    
    This would be {'greater than' if sum([3*values[i] for i in range(T)]) > w else 'less than'} your outside offer of {w}.

    The negotiation proceeds in {R} rounds.
    There is a discount rate gamma = {g}, such that if the process concludes after r rounds the overall value of the negotiation to each player is their value for the outcome multiplied by gamma to the power r-1 where r is the round number.
    At each round, Player 1 takes an action, followed by Player 2.
    The possible actions are to ACCEPT the other player's current offer (if any), make a COUNTEROFFER, or WALK away.  If the game gets to the last round, and player 2 chooses to make a counteroffer, this is treated as a WALK.
    If a player chooses ACCEPT, the negotiation ends in a deal to divide the items according to the accepted offer.
    Each player receives a value for this outcome determined by their private values per unit of each item, and the units they get in the deal.
    If a player chooses WALK, the negotiation ends without a deal, and each player receives the value of their private outside offer.

    Let's solve this step by step:

    1) First, let's analyze the current situation:
    - What is my outside offer value? 
    - What are my item values?
    - What is the current offer (if any)?
    - What round are we in and what is the discount factor?
    {f"- If I am considering making a counteroffer, what is the next round's discount factor (as that counteroffer will be realized in the next round, hence the discount factor will be gamma^(r+1-1) as opposed to gamma^(r-1) for the current round)?" if my_player_num == 2 else ""}

    2) Then, let's calculate:
    - What is the current discount factor, gamma^(r-1)?
    - For the current offer (if any): What would be my total value if I accept?
    - For potential counteroffers: What would be my total value for different divisions?
    - How do these compare to my outside offer value?
    - How does the discount factor affect these values?

    3) Let's decide:
    - Should I accept the current offer?
    - Should I walk away with my outside offer?
    - Or should I make a specific COUNTEROFFER?

    Please show your reasoning step by step, then provide your action in one of these formats in your response (if you do not do this your response will be invalid and treated as a WALK):
    {{"action": "ACCEPT"}} - to accept the current offer
    {{"action": "WALK"}} - to walk away from negotiations  
    {{"action": "COUNTEROFFER", "offer": [n1, n2, ...]}} - where n1, n2, ... are numbers representing your counteroffer

    Any response not in these exact formats will be invalid and treated as a WALK. If you provide a counteroffer, it must be a valid offer, otherwise it will be treated as a WALK.

    It is now round {r}.

    Negotiation history:{history_str}
    {current_offer_str}
    {action_prompt}
"""


