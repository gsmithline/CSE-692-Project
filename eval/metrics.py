
import itertools
import numpy as np
from math import sqrt, prod


def compute_pareto_frontier(p1_values, p2_values, num_items, items, outside_offer_values    ):
    
    all_allocations = itertools.product(
            *[range(int(items[i].item()) + 1) for i in range(num_items)]
    )
    allocations_with_values = []
    for allocation in all_allocations:
        agent1_value = 0.0
        agent2_value = 0.0
        for i, amount_for_agent1 in enumerate(allocation):
            amount_for_agent2 = int(items[i].item()) - amount_for_agent1
            agent1_value += amount_for_agent1 * float(p1_values[i].item())
            agent2_value += amount_for_agent2 * float(p2_values[i].item())
        allocations_with_values.append((allocation, agent1_value, agent2_value))
    allocations_with_values.append(((0, 0, 0, 0, 0), outside_offer_values[0], outside_offer_values[1])) #add outside offer to the allocations
    frontier_allocations = []
    for i, (alloc_i, val1_i, val2_i) in enumerate(allocations_with_values):
        dominated = False
        for j, (alloc_j, val1_j, val2_j) in enumerate(allocations_with_values):
            if j != i:
                if (val1_j >= val1_i and val2_j >= val2_i) and (val1_j > val1_i or val2_j > val2_i):
                    dominated = True
                    break
        if not dominated:
            frontier_allocations.append(alloc_i)
        
    result = []
    for alloc in frontier_allocations:
        agent1_split = list(alloc)
        agent2_split = [
            int(items[i].item()) - agent1_split[i] for i in range(num_items)
        ]
        result.append(
            {
                    "agent1": agent1_split,
                    "agent2": agent2_split
            }
        )
    
    return result

def compute_utilitarian_welfare(offer, values, gamma, realization_round):
    if offer is None:
        return 0
    base_value = sum(v * q for v, q in zip(values, offer))
    return base_value * (gamma ** (realization_round - 1))

def compute_raw_welfare(offer, values): #this is ex-post
    return sum(v * q for v, q in zip(values, offer))



def compute_security_level(offer, values, outside_offer_value): #this is ex-post
    compute_value = np.dot(offer, values)
    return max(0, outside_offer_value - compute_value)

def compute_gini_coefficient(offer, values): #this is ex-post
    return 1 - np.sum(np.square(offer / np.sum(offer)))

def compute_average_concession_size(offers, valuations): #this is ex-interim
    """
    Compute the average concession size (in utility terms) for a single player's sequence of offers.
    
    Parameters:
        offers (list of lists/tuples/tensors): 
            Each element is an allocation for the player, e.g. how many units of each item 
            they keep (or how many units they're offering to the opponent—just be consistent).
        valuations (list/tuple/tensor): 
            The player's values per unit of each item (same length as one offer).
    
    Returns:
        float: The average utility concession between consecutive offers.
               0 if there is only one (or zero) offers, or if no positive concessions occur.
    """
    if len(offers) < 2:
        return 0.0  
    
    total_concession = 0.0
    step_count = 0
    
    for i in range(1, len(offers)):
        old_utility = sum(v * q for v, q in zip(valuations, offers[i-1]))
        new_utility = sum(v * q for v, q in zip(valuations, offers[i]))
        concession = old_utility - new_utility
        if concession > 0:
            total_concession += concession
        
        step_count += 1

    # Average concession size
    return total_concession / step_count if step_count > 0 else 0.0

def proposal_proportionality(offer, values, num_items): #this is ex-interim
    return np.dot(offer, values) / np.dot(num_items, values)
'''
call this expost regret 
Something like ex-post regret max(0, value of your final result - other offer on table)
'''


    

