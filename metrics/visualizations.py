import matplotlib.pyplot as plt
import numpy as np

def plot_discounted_values(rounds, p1_values, p2_values, max_rounds):
    plt.figure(figsize=(10, 6))
    
    # Determine the indices for P1 and P2 offers
    p1_offer_indices = [i for i in range(len(rounds)) if i % 2 == 0]  # P1's offers
    p2_offer_indices = [i for i in range(len(rounds)) if i % 2 == 1]  # P2's offers
    
    # Plot P1 offers at integer rounds
    p1_x = [rounds[i] for i in p1_offer_indices]
    plt.plot(p1_x, 
             [p1_values[i] for i in p1_offer_indices], 
             'b-o', label='P1 value from P1 offers', 
             markerfacecolor='white', markersize=10)
    
    plt.plot(p1_x, 
             [p2_values[i] for i in p1_offer_indices], 
             'r-o', label='P2 value from P1 offers', 
             markerfacecolor='white', markersize=10)
    
    # Plot P2 offers at x.5 rounds
    p2_x = [rounds[i] + 0.5 for i in p2_offer_indices]
    plt.plot(p2_x, 
             [p1_values[i] for i in p2_offer_indices], 
             'b-s', label='P1 value from P2 offers', 
             markerfacecolor='white', markersize=10)
    
    plt.plot(p2_x, 
             [p2_values[i] for i in p2_offer_indices], 
             'r-s', label='P2 value from P2 offers', 
             markerfacecolor='white', markersize=10)
    
    # Draw vertical lines to separate rounds
    for r in range(1, max_rounds + 1):
        plt.axvline(x=r - 0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Set x-ticks at both integer and half-integer positions
    xticks = []
    xticklabels = []
    for i in range(1, max_rounds + 1):
        xticks.extend([i - 0.5, i])
        xticklabels.extend(['', str(i)])
    plt.xticks(xticks, xticklabels)
    
    plt.xlabel('Round')
    plt.ylabel('Discounted Value')
    plt.title('Discounted Values by Offer Type\n(○: P1 Offers, □: P2 Counter-offers)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_offer_evolution(game, rounds, p1_offers, p2_offers):
    num_rows = (game.num_items + 1) // 2 
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, 3*num_rows))
    fig.suptitle('Item Distribution Over Rounds')
    
    # Debug print to understand our data
    print(f"Number of rounds: {len(rounds)}")
    print(f"Number of P1 offers: {len(p1_offers)}")
    print(f"Number of P2 offers: {len(p2_offers)}")
    
    # Use the actual number of rounds we have data for
    max_round = max(len(p1_offers), len(p2_offers))
    valid_rounds = range(1, max_round + 1)
    
    axs = np.array(axs).reshape(num_rows, 2)
    
    for i in range(game.num_items):
        row = i // 2
        col = i % 2
        
        p1_x = list(valid_rounds)
        p2_x = [r + 0.5 for r in valid_rounds[:len(p2_offers)]]
        
        axs[row, col].plot(p1_x[:len(p1_offers)], [h[i] for h in p1_offers], 'b-o', label='P1 Offers')
        axs[row, col].plot(p2_x, [h[i] for h in p2_offers], 'r-o', label='P2 Offers')
        axs[row, col].set_title(f'Item {i+1}')
        axs[row, col].set_xlabel('Round')
        axs[row, col].set_ylabel('Units')
        
        xticks = []
        xticklabels = []
        for r in range(1, max_round + 1):
            xticks.append(r)
            xticklabels.append(str(r))
            xticks.append(r + 0.5)
            xticklabels.append('')
        axs[row, col].set_xticks(xticks)
        axs[row, col].set_xticklabels(xticklabels)
        
        axs[row, col].set_yticks(range(0, game.items[i] + 1))
        axs[row, col].legend()
        axs[row, col].grid(True, alpha=0.3)
        
    if game.num_items % 2 == 1:
        fig.delaxes(axs[-1, -1])
        
    plt.tight_layout()
    plt.show()

def plot_negotiation_gap(rounds, p1_values, p2_values):
    plt.figure(figsize=(10, 6))
    
    # Split by who made the offer
    p1_offer_indices = [i for i in range(len(rounds)) if i % 2 == 0]  # P1's offers
    p2_offer_indices = [i for i in range(len(rounds)) if i % 2 == 1]  # P2's offers
    
    # Calculate gaps for P1's offers
    gaps_p1_offers = [(p1_values[i] - p2_values[i])/(p1_values[i] + p2_values[i]) 
                      for i in p1_offer_indices]
    plt.plot([r + 0.5 for r in rounds[::2]], gaps_p1_offers, 'g-o', 
             label='Value Gap (P1 offers)', markerfacecolor='white')
    
    # Calculate gaps for P2's offers
    gaps_p2_offers = [(p1_values[i] - p2_values[i])/(p1_values[i] + p2_values[i]) 
                      for i in p2_offer_indices]
    plt.plot(rounds[1::2], gaps_p2_offers, 'g-s', 
             label='Value Gap (P2 offers)', markerfacecolor='white')
    
    plt.axhline(y=0, color='r', linestyle='--', label='Equal Value')
    plt.xlabel('Round')
    plt.ylabel('Normalized Value Difference (P1 - P2)/(P1 + P2)')
    plt.title('Normalized Negotiation Gap Over Time\n(○: P1 Offers, □: P2 Counter-offers)')
    plt.xticks([i + 0.5 for i in range(0, max(rounds)+1)], [f"{i+1}" if i % 1 == 0 else f"{i+0.5}" for i in range(0, max(rounds)+1)])
    plt.legend()
    plt.grid(True)
    plt.show()
    return gaps_p1_offers, gaps_p2_offers
''''
def plot_negotiation_gap(rounds, p1_values, p2_values):
    plt.figure(figsize=(10, 6))
    # Normalize by total value in each round
    value_gaps = [(p1 - p2)/(p1 + p2) for p1, p2 in zip(p1_values, p2_values)]
    
    plt.plot(rounds, value_gaps, 'g-o', label='Normalized Value Gap')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Round')
    plt.ylabel('Normalized Value Difference (P1 - P2)/(P1 + P2)')
    plt.title('Normalized Negotiation Gap Over Time')
    plt.xticks(range(1, max(rounds)+1))
    plt.legend()
    plt.grid(True)
    plt.show()
    return value_gaps
'''
'''
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
'''
def plot_fairness(rounds, p1_values, p2_values):
    plt.figure(figsize=(10, 6))
    
    p1_offer_indices = [i for i in range(len(rounds)) if i % 2 == 0]  
    p2_offer_indices = [i for i in range(len(rounds)) if i % 2 == 1]  
    
  
    fairness_p1_offers = [(p1_values[i]/(p1_values[i]+p2_values[i]), 
                          p2_values[i]/(p1_values[i]+p2_values[i])) 
                         for i in p1_offer_indices]
    plt.plot(rounds[::2], [f[0] for f in fairness_p1_offers], 'b-o', 
             label='P1 Share (P1 offers)', markerfacecolor='white')
    plt.plot(rounds[::2], [f[1] for f in fairness_p1_offers], 'r-o', 
             label='P2 Share (P1 offers)', markerfacecolor='white')
    
    
    fairness_p2_offers = [(p1_values[i]/(p1_values[i]+p2_values[i]), 
                          p2_values[i]/(p1_values[i]+p2_values[i])) 
                         for i in p2_offer_indices]
    plt.plot(rounds[1::2], [f[0] for f in fairness_p2_offers], 'b-s', 
             label='P1 Share (P2 offers)', markerfacecolor='white')
    plt.plot(rounds[1::2], [f[1] for f in fairness_p2_offers], 'r-s', 
             label='P2 Share (P2 offers)', markerfacecolor='white')
    
    plt.axhline(y=0.5, color='g', linestyle='--', label='Equal Split')
    plt.ylabel('Proportion of Total Value')
    plt.xticks(range(1, max(rounds)+1))
    plt.title('Value Distribution Fairness\n(○: P1 Offers, □: P2 Counter-offers)')
    plt.legend()
    plt.grid(True)
    plt.show()

