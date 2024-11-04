from dataclasses import dataclass

@dataclass
class Offer:
    player: int
    offer: list[int] | bool
