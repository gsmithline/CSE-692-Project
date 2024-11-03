def make_prompt(T: int, quantities: list[int], V: int, values: list[float], W: int, w: int, R: int, g: float, r: int) -> str:
    return f"""
    You and another agent have to negotiate a division of items between the two of you.
    Let us call you Player 1 and the other agent Player 2.
    There are {T} types of items, called item 1 through item {T}.
    There are {', '.join([f'{q} units of item {i+1}' for i, q in enumerate(quantities)])} to divide.
    Both you and Player 2 have a private value per unit of each item type.
    These values are drawn from a uniform random distribution, ranging from 0 to {V}.
    Your private values are {', '.join([f'{v} for item {i+1}' for i, v in enumerate(values)])}.
    Both you and Player 2 also have a private outside offer, of value drawn from a uniform random distribution, ranging from 0 to {W}.
    Your outside offer value is {w}.

    The negotiation proceeds in {R} rounds.
    There is a discount rate gamma = {g}, such that if the process concludes after r rounds the overall value of the negotiation to each player is their value for the outcome multiplied by gamma to the power r-1.
    At each round, Player 1 takes an action, followed by Player 2.
    The possible actions are to ACCEPT the other player's current offer (if any), make a new OFFER, or WALK away.
    If a player chooses ACCEPT, the negotiation ends in a deal to divide the items according to the accepted offer.
    Each player receives a value for this outcome determined by their private values per unit of each item, and the units they get in the deal.
    If a player chooses WALK, the negotiation ends without a deal, and each player receives the value of their private outside offer.
    Otherwise, the player chooses to provide a new offer for how to divide the items, and the negotiation proceeds.
    The offer specifies for each item type an integer number of units the player proposes to keep for themself, up to the number of units available. Under this offer, the other player would get the remaining units.

    It is now round {r}.
    """
