
import itertools
import numpy as np
from math import sqrt, prod


import itertools

def compute_pareto_frontier(p1_values, p2_values, num_items, items, outside_offer_values):
    # Enumerate all integer item allocations
    all_allocations = itertools.product(
        *[range(int(items[i].item()) + 1) for i in range(num_items)]
    )
    allocations_with_values = []

    # 1) Compute the payoff (val1, val2) for each item allocation
    for allocation in all_allocations:
        agent1_value = 0.0
        agent2_value = 0.0
        for i, amount_for_agent1 in enumerate(allocation):
            amount_for_agent2 = int(items[i].item()) - amount_for_agent1
            agent1_value += amount_for_agent1 * float(p1_values[i].item())
            agent2_value += amount_for_agent2 * float(p2_values[i].item())

        allocations_with_values.append((allocation, agent1_value, agent2_value))

    allocations_with_values.append(
        ("OUTSIDE_OFFER", outside_offer_values[0], outside_offer_values[1])
    )

    frontier_allocations = []
    for i, (alloc_i, val1_i, val2_i) in enumerate(allocations_with_values):
        dominated = False
        for j, (alloc_j, val1_j, val2_j) in enumerate(allocations_with_values):
            if j != i:
                # (val1_j, val2_j) >= (val1_i, val2_i) and strictly >
                if (val1_j >= val1_i and val2_j >= val2_i) \
                   and (val1_j > val1_i or val2_j > val2_i):
                    dominated = True
                    break
        if not dominated:
            frontier_allocations.append(alloc_i)

    result = []
    for alloc in frontier_allocations:
        if alloc == "OUTSIDE_OFFER":
            result.append({
                "type": "outside_offer",
                "agent1_value": outside_offer_values[0],
                "agent2_value": outside_offer_values[1]
            })
        else:
            # Normal item allocation
            agent1_split = list(alloc)
            agent2_split = [
                int(items[i].item()) - agent1_split[i] for i in range(num_items)
            ]
            result.append({
                "type": "allocation",
                "agent1": agent1_split,
                "agent2": agent2_split
            })

    return result


def compute_utilitarian_welfare(offer, values, gamma, realization_round):
    if offer is None:
        return 0
    base_value = sum(v * q for v, q in zip(values, offer))
    return base_value * (gamma ** (realization_round - 1))

def compute_raw_welfare(offer, values): #this is ex-post
    return sum(v * q for v, q in zip(values, offer))



    

