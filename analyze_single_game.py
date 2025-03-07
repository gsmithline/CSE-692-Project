import json

def parse_values(line):
    """Parse the line containing private values"""
    # Format: "Your private values are 16 for item 1, 46 for item 2, 100 for item 3, 23 for item 4, 47 for item 5."
    values = []
    parts = line.split("are ")[1].split(", ")
    for part in parts:
        value = int(part.split(" for item")[0])
        values.append(value)
    return values

def parse_outside_offer(line):
    """Parse the line containing outside offer"""
    # Format: "Your outside offer value is 145. Your objective..."
    return int(line.split("value is ")[1].split(".")[0])

def calculate_offer_value(offer, values, num_items):
    """Calculate value of an offer given item values and total items available"""
    value = 0
    for i in range(len(offer)):
        items_kept = num_items[i] - offer[i]  # Calculate items kept by the player
        value += items_kept * values[i]
    return value

def analyze_single_game(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = []
    for game in data['all_game_data']:
        agent1 = game['agent1']
        agent2 = game['agent2']
        
        # Track the final state
        final_action = None
        final_offer = None
        p1_outside_offer = None
        p2_outside_offer = None
        p1_values = None
        p2_values = None
        num_items = None
        
        for round_data in game['round_data']:
            final_action = round_data['action']
            
            # Extract values from the prompt
            prompt = round_data['prompt']
            if "You are Player 1" in prompt:
                # Extract Player 1's values and outside offer
                for line in prompt.split('\n'):
                    if "Your private values are" in line:
                        p1_values = parse_values(line)
                    elif "Your outside offer value is" in line:
                        p1_outside_offer = parse_outside_offer(line)
                    elif "There are" in line and "units of item" in line:
                        if num_items is None:
                            # Extract numbers before "units of item"
                            nums = []
                            parts = line.split("There are ")[1].split(", ")
                            for part in parts:
                                num = int(part.split(" units")[0])
                                nums.append(num)
                            num_items = nums
            elif "You are Player 2" in prompt:
                # Extract Player 2's values and outside offer
                for line in prompt.split('\n'):
                    if "Your private values are" in line:
                        p2_values = parse_values(line)
                    elif "Your outside offer value is" in line:
                        p2_outside_offer = parse_outside_offer(line)
            
            # Track the current offer
            if "Current offer on the table" in prompt:
                offer_line = prompt.split("Current offer on the table")[1].split("\n")[0]
                if "None" not in offer_line and "[" in offer_line:
                    final_offer = eval(offer_line[offer_line.find("["):offer_line.find("]")+1])
        
        # Calculate final values
        p1_final_value = None
        p2_final_value = None
        
        if final_action == "WALK":
            p1_final_value = p1_outside_offer
            p2_final_value = p2_outside_offer
        elif final_action == "ACCEPT":
            # The final offer represents what's given to Player 2
            if final_offer:
                p2_final_value = calculate_offer_value(final_offer, p2_values, num_items)
                p1_final_value = calculate_offer_value([num_items[i] - final_offer[i] for i in range(len(final_offer))], p1_values, num_items)
        
        results.append({
            'agent1': agent1,
            'agent2': agent2,
            'final_action': final_action,
            'agent1_value': p1_final_value,
            'agent2_value': p2_final_value,
            'circle_values': data['all_game_data'][0]['circle']
        })
    
    return results

# Example usage:
file_path = "crossplay/sonnet_3.7_gemini_circle_6_vs_circle_2/all_game_data_3_1_2025_20_anthropic_3.7_sonnet_vs_gemini_2.0_flash_circle_p1_6_circle_p2_2.json"
results = analyze_single_game(file_path)

# Print results for this file
for i, game in enumerate(results):
    print(f"\nGame {i+1} between {game['agent1']} and {game['agent2']}")
    print(f"Circle values: {game['circle_values']}")
    print(f"Final action: {game['final_action']}")
    print(f"Final values: Agent1: {game['agent1_value']}, Agent2: {game['agent2_value']}")