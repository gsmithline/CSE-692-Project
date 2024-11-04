import matplotlib.pyplot as plt
import numpy as np

def plot_discounted_values(rounds, p1_values, p2_values, max_rounds):
    plt.figure(figsize=(10, 6))
    
    # Separate values by who made the offer
    p1_offer_indices = [i for i in range(len(rounds)) if i % 2 == 0]  # P1's offers
    p2_offer_indices = [i for i in range(len(rounds)) if i % 2 == 1]  # P2's offers
    
    # Plot 4 separate lines:
    # Value to P1 from P1's offers (blue circles)
    plt.plot([rounds[i] for i in p1_offer_indices], 
             [p1_values[i] for i in p1_offer_indices], 
             'b-o', label='P1 value from P1 offers', 
             markerfacecolor='white', markersize=10)
    
    # Value to P2 from P1's offers (red circles)
    plt.plot([rounds[i] for i in p1_offer_indices], 
             [p2_values[i] for i in p1_offer_indices], 
             'r-o', label='P2 value from P1 offers', 
             markerfacecolor='white', markersize=10)
    
    # Value to P1 from P2's offers (blue squares)
    plt.plot([rounds[i] for i in p2_offer_indices], 
             [p1_values[i] for i in p2_offer_indices], 
             'b-s', label='P1 value from P2 offers', 
             markerfacecolor='white', markersize=10)
    
    # Value to P2 from P2's offers (red squares)
    plt.plot([rounds[i] for i in p2_offer_indices], 
             [p2_values[i] for i in p2_offer_indices], 
             'r-s', label='P2 value from P2 offers', 
             markerfacecolor='white', markersize=10)
    
    # Add vertical lines to separate rounds
    for r in range(1, max(rounds)):
        plt.axvline(x=r+0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Customize the plot
    plt.xlabel('Round')
    plt.xticks(range(1, max(rounds)+1))
    plt.ylabel('Discounted Value')
    plt.title('Discounted Values by Offer Type\n(○: P1 Offers, □: P2 Counter-offers)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()

def plot_offer_evolution(game, rounds, p1_offers, p2_offers):
    num_rows = (game.num_items + 1) // 2 
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, 3*num_rows))
    fig.suptitle('Item Distribution Over Rounds')
    
    valid_rounds = rounds[:len(p1_offers)]
    for i in range(game.num_items):
        row = i // 2
        col = i % 2
        axs[row, col].plot(valid_rounds, [h[i] for h in p1_offers], 'b-o', label='P1 Offers')
        axs[row, col].plot(valid_rounds, [h[i] for h in p2_offers], 'r-o', label='P2 Offers')
        axs[row, col].set_title(f'Item {i+1}')
        axs[row, col].set_xlabel('Round')
        axs[row, col].set_ylabel('Units')
        axs[row, col].set_xticks(range(1, max(valid_rounds) + 1))
        axs[row, col].set_yticks(range(0, game.items[i] + 1))
        axs[row, col].legend()
    
    if game.num_items % 2 == 1:
        axs[-1, -1].remove()
    plt.tight_layout()

def plot_negotiation_gap(rounds, p1_values, p2_values):
    plt.figure(figsize=(10, 6))
    value_gaps = [p1 - p2 for p1, p2 in zip(p1_values, p2_values)]
    plt.plot(rounds, value_gaps, 'g-o', label='Value Gap')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Round')
    plt.ylabel('Value Difference (P1 - P2)')
    plt.title('Negotiation Gap Over Time')
    plt.xticks(range(1, max(rounds)+1))
    plt.legend()
    plt.grid(True)
    plt.show()
    return value_gaps

def plot_fairness(rounds, p1_values, p2_values):
    plt.figure(figsize=(10, 6))
    fairness = [(p1/(p1+p2), p2/(p1+p2)) for p1, p2 in zip(p1_values, p2_values)]
    plt.plot(rounds, [f[0] for f in fairness], 'b-o', label='P1 Share')
    plt.plot(rounds, [f[1] for f in fairness], 'r-o', label='P2 Share')
    plt.axhline(y=0.5, color='g', linestyle='--', label='Equal Split')
    plt.ylabel('Proportion of Total Value')
    plt.xticks(range(1, max(rounds)+1))
    plt.title('Value Distribution Fairness')
    plt.legend()
    plt.grid(True)
    plt.show()
    return fairness