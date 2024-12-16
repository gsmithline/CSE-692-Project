from utils.offer import Offer

def make_prompt_bargain(T: int, quantities: list[int], V: int, values: list[float], W1: int, W2: int, w: int, R: int, g: float, r: int, history: dict, current_offer: Offer = None, player_num: int = 0) -> str:
    
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

    current_offer_str = f"\nCurrent offer on the table: {current_offer.offer if current_offer else 'None'}"
    
    
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
    You are a bargaining agent.  You and another bargaining agent have to bargain over an allocation of items between the two of you.
    You are Player {my_player_num} and the other agent is Player {other_player_num}.
    There are {T} types of items, called item 1 through item {T}.
    There are {', '.join([f'{q} units of item {i+1}' for i, q in enumerate(quantities)])} to allocate.
    Both you and Player {other_player_num} have a private value per unit of each item type.
    These values are drawn from a uniform random distribution, ranging from 0 to {V}.
    Your private values are {', '.join([f'{v} for item {i+1}' for i, v in enumerate(values)])}.
    Both you and Player {other_player_num} also have a private outside offer, of value drawn from a uniform random distribution, ranging from {W2} to {W1}.
    Your outside offer value is {w}. Your objective is to maximize your value of the outcome of the bargaining game.  Remember, you have a guaranteed alternative: your outside offer. Always consider when accepting or proposing an offer whether that offer has value greater than or equal to your outside offer.
    For example, if you were to keep 1 unit of item 1 with value {values[0]} per unit, 1 unit of item 2 with value {values[1]} per unit, 1 unit of item 3 with value {values[2]} per unit, 1 unit of item 4 with value {values[3]} per unit, this would give you a total value of {sum([values[i] for i in range(T)])}, which you can compare to your outside offer value of {w}.

    The bargaining game proceeds in {R} rounds.
    There is a discount rate gamma = {g}, such that if the bargaining process concludes after r rounds the overall value of the allocation to each player is their value for the outcome multiplied by gamma to the power r-1 where r is the round number.
    At each round, Player 1 takes an action, followed by Player 2.
    The possible actions are to ACCEPT the other player's current offer (if any), make a COUNTEROFFER, or WALK away. If the game gets to the last round, and player 2 chooses to make a counteroffer, this is treated as a WALK.
    If a player chooses ACCEPT, the bargaining ends in a deal to allocate the items according to the accepted allocation.
    Each player receives a value for this outcome determined by their private values per unit of each item, and the units they get in the deal.
    If a player chooses WALK, the bargaining ends without a deal, and each player receives the value of their private outside offer.
    The action you decide out of these three actions is all you should provide in your response.
    Otherwise, the player chooses to provide a counteroffer for how to allocate the items, and the bargaining proceeds.

    It is now round {r}.

    Bargaining history:{history_str}
    {current_offer_str}
    {action_prompt}
    """


#Your objective is to maximize your value of the outcome of the bargaining game. Remember, you have a guaranteed alternative: your outside offer. Always consider when accepting or proposing an offer whether that offer has value greater than or equal to your outside offer.
   # For example, if you were to keep {quantities[0]} units of item 1 with value {values[0]} per unit, this would give you a total value of {quantities[0] * values[0]}, which you can compare to your outside offer value of {w}.