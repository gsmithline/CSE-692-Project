import json
import re
import math
import matplotlib.pyplot as plt
import numpy as np

DISCOUNT_RATE = 0.9

def get_discount_factor(round_index, player_id):
    exponent = (round_index - 1) if player_id == 1 else round_index
    return DISCOUNT_RATE ** exponent

def parse_valuations_from_prompt(prompt_text):
    player_id = None
    if "You are Player 1" in prompt_text:
        player_id = 1
    elif "You are Player 2" in prompt_text:
        player_id = 2
    else:
        raise ValueError("Could not determine player (Player 1 or Player 2) from prompt.")

    item_qty_pattern = re.compile(
        r"There\s+are\s+(\d+)\s+units\s+of\s+item\s+1,\s+(\d+)\s+units\s+of\s+item\s+2,\s+(\d+)\s+units\s+of\s+item\s+3,\s+(\d+)\s+units\s+of\s+item\s+4,\s+(\d+)\s+units\s+of\s+item\s+5"
    )
    item_qty_match = item_qty_pattern.search(prompt_text.replace("\n", " "))
    if not item_qty_match:
        raise ValueError("Could not parse item quantities from prompt.")
    items = list(map(int, item_qty_match.groups()))

    val_pattern = re.compile(
        r"Your\s+private\s+values\s+are\s+(\d+)\s+for\s+item\s+1,\s+(\d+)\s+for\s+item\s+2,\s+(\d+)\s+for\s+item\s+3,\s+(\d+)\s+for\s+item\s+4,\s+(\d+)\s+for\s+item\s+5"
    )
    val_match = val_pattern.search(prompt_text.replace("\n", " "))
    if not val_match:
        raise ValueError("Could not parse valuations from prompt.")
    valuations = list(map(int, val_match.groups()))

    outoffer_pattern = re.compile(r"Your\s+outside\s+offer\s+value\s+is\s+(\d+)")
    outoffer_match = outoffer_pattern.search(prompt_text.replace("\n", " "))
    if not outoffer_match:
        raise ValueError("Could not parse outside offer from prompt.")
    outside_offer = int(outoffer_match.group(1))

    return {
        "player_id": player_id,
        "valuations": valuations,
        "items": items,
        "outside_offer": outside_offer
    }

def parse_offer_from_response(response_text):
    pattern = re.compile(r"\"offer\"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")
    match = pattern.search(response_text)
    return list(map(int, match.groups())) if match else None

def compute_allocation_value(valuations, allocation):
    if valuations is None or allocation is None:
        return 0
    elif any(x is None for x in valuations) or any(x is None for x in allocation):
        return 0
    return sum(v * a for v, a in zip(valuations, allocation))

def detect_mistakes(current_move_info, previous_move_info_player, opponent_last_offer_info=None):
    mistakes_found = []
    action = current_move_info["action"]
    my_val_cur = current_move_info["my_value_current_offer"]
    my_val_prev = current_move_info["my_value_previous_own_offer"]
    out_offer = current_move_info["my_outside_offer"]
    allocation_to_opp = current_move_info["allocation_offered_to_opponent"]
    allocation_i_keep = current_move_info["allocation_i_keep"]
    is_game_ending = current_move_info["is_game_ending"]
    
    if action == "COUNTEROFFER" and not is_game_ending:
        # Mistake 1: Making an offer worse than opponent's offer on the table
        if opponent_last_offer_info is not None:
            value_of_opponents_offer = opponent_last_offer_info
            if my_val_cur < value_of_opponents_offer:
                mistakes_found.append(1)
        
        # Mistake 2: Making an offer worse than outside offer
        if my_val_cur < out_offer:
            mistakes_found.append(2)
            
        # Mistake 3: Offering nothing or everything
        if allocation_to_opp is not None and allocation_i_keep is not None:
            sum_offered = sum(allocation_to_opp)
            sum_kept = sum(allocation_i_keep)
            total_items = sum_offered + sum_kept
            if total_items > 0:
                if sum_offered == 0 or sum_offered == total_items:
                    mistakes_found.append(3)

    if action == "ACCEPT" and is_game_ending:
        # Mistake 4: Accepting an offer worse than outside offer
        if my_val_cur < out_offer:
            mistakes_found.append(4)

    if action == "WALK" and is_game_ending:
        # Mistake 5: Walking away from an offer better than outside offer
        if my_val_cur > out_offer:
            mistakes_found.append(5)

    return mistakes_found

def analyze_circle6(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_game_data = data["all_game_data"]
    num_games = len(all_game_data)
    num_moves = 0
    moves_by_round = {1: 0, 2: 0, 3: 0}
    moves_by_player = {1: 0, 2: 0}
    mistake_counts = {k: 0 for k in range(1, 6)}
    mistake_counts_by_player = {
        1: {k: 0 for k in range(1, 6)},
        2: {k: 0 for k in range(1, 6)}
    }
    mistake_counts_by_round = {
        1: {k: 0 for k in range(1, 6)},
        2: {k: 0 for k in range(1, 6)},
        3: {k: 0 for k in range(1, 6)},
    }

    for game in all_game_data:
        round_data = game["round_data"]
        last_move_info = {1: None, 2: None}

        opponent_offer_values = {1: None, 2: None}
        total_rounds = 3
        current_round = 1
        i = 0
        while i < len(round_data) and current_round <= total_rounds:
            # ---------------------------------------
            # Process the first move of this round
            # ---------------------------------------
            first_move = round_data[i]
            i += 1

            try:
                prompt = first_move["prompt"]
                response = first_move["response"]
                action = first_move["action"]
            except KeyError:
                current_round += 1
                continue

            try:
                parsed_info = parse_valuations_from_prompt(prompt)
            except ValueError:
                current_round += 1
                continue

            player_id = parsed_info["player_id"]
            opponent_id = 2 if player_id == 1 else 1
            valuations = parsed_info["valuations"]
            items = parsed_info["items"]
            outside_offer = parsed_info["outside_offer"]
            discount_factor = get_discount_factor(current_round, player_id)
            discounted_outside_offer = outside_offer * discount_factor
            is_game_ending = (action in ["ACCEPT", "WALK", "INVALID WALK"])
            offer_allocation = parse_offer_from_response(response) if action == "COUNTEROFFER" else None
            
            if action == "COUNTEROFFER":
                if offer_allocation is not None:
                    allocation_offered_to_opponent = offer_allocation
                    allocation_i_keep = [ti - oa for ti, oa in zip(items, offer_allocation)]
                    current_value_for_self = compute_allocation_value(valuations, allocation_i_keep) * discount_factor
                else:
                    allocation_offered_to_opponent = None
                    allocation_i_keep = None
                    current_value_for_self = 0.0
            elif action == "ACCEPT":
                opp_id = 1 if player_id == 2 else 2
                opp_info = last_move_info[opp_id]
                if opp_info and opp_info["action"] == "COUNTEROFFER":
                    our_alloc = opp_info["allocation_offered_to_opponent"]
                    current_value_for_self = compute_allocation_value(valuations, our_alloc) * discount_factor
                    allocation_offered_to_opponent = None
                    allocation_i_keep = our_alloc
                else:
                    current_value_for_self = 0.0
                    allocation_offered_to_opponent = None
                    allocation_i_keep = None
            elif action in ["WALK", "INVALID WALK"]:
                opp_id = 1 if player_id == 2 else 2
                opp_info = last_move_info[opp_id]
                if opp_info and opp_info["action"] == "COUNTEROFFER":
                    our_alloc = opp_info["allocation_offered_to_opponent"]
                    current_value_for_self = compute_allocation_value(valuations, our_alloc) * discount_factor
                    allocation_offered_to_opponent = None
                    allocation_i_keep = our_alloc
                else:
                    current_value_for_self = 0.0
                    allocation_offered_to_opponent = None
                    allocation_i_keep = None
            else:
                continue

            my_value_previous_own_offer = None
            if last_move_info[player_id] and last_move_info[player_id]["action"] == "COUNTEROFFER":
                my_value_previous_own_offer = last_move_info[player_id]["my_value_current_offer"]

            opponent_value_for_me = opponent_offer_values.get(player_id)

            current_move_info_struct = {
                "player_id": player_id,
                "action": action,
                "my_value_current_offer": current_value_for_self,
                "my_outside_offer": discounted_outside_offer,
                "my_value_previous_own_offer": my_value_previous_own_offer,
                "allocation_offered_to_opponent": allocation_offered_to_opponent,
                "allocation_i_keep": allocation_i_keep,
                "is_game_ending": is_game_ending,
            }

            mistakes_triggered = detect_mistakes(
                current_move_info_struct,
                last_move_info.get(player_id),
                opponent_value_for_me
            )
            
            for mk in mistakes_triggered:
                mistake_counts[mk] += 1
                mistake_counts_by_player[player_id][mk] += 1
                mistake_counts_by_round[current_round][mk] += 1

            if action in ["COUNTEROFFER", "ACCEPT", "WALK", "INVALID WALK"]:
                num_moves += 1
                moves_by_round[current_round] += 1
                moves_by_player[player_id] += 1

            if action == "COUNTEROFFER":
                # Update the offer this player made for the opponent to track
                last_move_info[player_id] = {
                    "action": action,
                    "my_value_current_offer": current_value_for_self,
                    "allocation_offered_to_opponent": allocation_offered_to_opponent,
                    "allocation_i_keep": allocation_i_keep
                }
                
                # Store value of this offer for the opponent
                opp_value = compute_allocation_value(valuations, allocation_offered_to_opponent) * discount_factor
                opponent_offer_values[opponent_id] = opp_value
                
            if is_game_ending:
                current_round += 1
                break

            # ---------------------------------------
            # Process the second move of this round (if any)
            # ---------------------------------------
            if i < len(round_data):
                second_move = round_data[i]
                i += 1

                try:
                    prompt2 = second_move["prompt"]
                    response2 = second_move["response"]
                    action2 = second_move["action"]
                except KeyError:
                    current_round += 1
                    continue

                try:
                    parsed_info2 = parse_valuations_from_prompt(prompt2)
                except ValueError:
                    current_round += 1
                    continue

                player_id2 = parsed_info2["player_id"]
                opponent_id2 = 2 if player_id2 == 1 else 1
                valuations2 = parsed_info2["valuations"]
                items2 = parsed_info2["items"]
                outside_offer2 = parsed_info2["outside_offer"]
                
                discount_factor2 = get_discount_factor(current_round, player_id2)
                discounted_outside_offer2 = outside_offer2 * discount_factor2
                is_game_ending2 = (action2 in ["ACCEPT", "WALK", "INVALID WALK"])

                offer_allocation2 = parse_offer_from_response(response2) if action2 == "COUNTEROFFER" else None
                
                if action2 == "COUNTEROFFER":
                    if offer_allocation2 is not None:
                        allocation_offered_to_opponent2 = offer_allocation2
                        allocation_i_keep2 = [ti - oa for ti, oa in zip(items2, offer_allocation2)]
                        current_value_for_self2 = compute_allocation_value(valuations2, allocation_i_keep2) * discount_factor2
                    else:
                        allocation_offered_to_opponent2 = None
                        allocation_i_keep2 = None
                        current_value_for_self2 = 0.0
                elif action2 == "ACCEPT":
                    opp_id2 = 1 if player_id2 == 2 else 2
                    opp_info2 = last_move_info[opp_id2]
                    if opp_info2 and opp_info2["action"] == "COUNTEROFFER":
                        our_alloc2 = opp_info2["allocation_offered_to_opponent"]
                        try:
                            current_value_for_self2 = compute_allocation_value(valuations2, our_alloc2) * discount_factor2
                        except:
                            current_value_for_self2 = 0.0
                        allocation_offered_to_opponent2 = None
                        allocation_i_keep2 = our_alloc2
                    else:
                        current_value_for_self2 = 0.0
                        allocation_offered_to_opponent2 = None
                        allocation_i_keep2 = None 
                elif action2 in ["WALK", "INVALID WALK"]:
                    opp_id2 = 1 if player_id2 == 2 else 2
                    opp_info2 = last_move_info[opp_id2]
                    if opp_info2 and opp_info2["action"] == "COUNTEROFFER":
                        our_alloc2 = opp_info2["allocation_offered_to_opponent"]
                        try:
                            current_value_for_self2 = compute_allocation_value(valuations2, our_alloc2) * discount_factor2
                        except:
                            current_value_for_self2 = 0.0
                        allocation_offered_to_opponent2 = None
                        allocation_i_keep2 = our_alloc2
                    else:
                        current_value_for_self2 = 0.0
                        allocation_offered_to_opponent2 = None
                        allocation_i_keep2 = None
                else:
                    current_round += 1
                    continue

                my_value_previous_own_offer2 = None
                if last_move_info[player_id2] and last_move_info[player_id2]["action"] == "COUNTEROFFER":
                    my_value_previous_own_offer2 = last_move_info[player_id2]["my_value_current_offer"]

                opponent_value_for_me2 = opponent_offer_values.get(player_id2)

                current_move_info_struct2 = {
                    "player_id": player_id2,
                    "action": action2,
                    "my_value_current_offer": current_value_for_self2,
                    "my_outside_offer": discounted_outside_offer2,
                    "my_value_previous_own_offer": my_value_previous_own_offer2,
                    "allocation_offered_to_opponent": allocation_offered_to_opponent2,
                    "allocation_i_keep": allocation_i_keep2,
                    "is_game_ending": is_game_ending2,
                }

                mistakes_triggered2 = detect_mistakes(
                    current_move_info_struct2,
                    last_move_info.get(player_id2),
                    opponent_value_for_me2
                )
                
                for mk2 in mistakes_triggered2:
                    mistake_counts[mk2] += 1
                    mistake_counts_by_player[player_id2][mk2] += 1
                    mistake_counts_by_round[current_round][mk2] += 1

                if action2 in ["COUNTEROFFER", "ACCEPT", "WALK", "INVALID WALK"]:
                    num_moves += 1
                    moves_by_round[current_round] += 1
                    moves_by_player[player_id2] += 1

                if action2 == "COUNTEROFFER":
                    last_move_info[player_id2] = {
                        "action": action2,
                        "my_value_current_offer": current_value_for_self2,
                        "allocation_offered_to_opponent": allocation_offered_to_opponent2,
                        "allocation_i_keep": allocation_i_keep2
                    }
                    
                    opp_value2 = compute_allocation_value(valuations2, allocation_offered_to_opponent2) * discount_factor2
                    opponent_offer_values[opponent_id2] = opp_value2
                    
                if is_game_ending2:
                    current_round += 1
                    break
            else:
                pass

            current_round += 1

    def ratio_str(numerator, denominator):
        if denominator == 0:
            return "0/0 (0.0%)"
        perc = (numerator / denominator) * 100.0
        return f"{numerator}/{denominator} ({perc:.1f}%)"

    def print_mistake_results(mk):
        if mk in (4, 5):
            overall_denom = num_games
        else:
            overall_denom = num_moves
        print(f"=== Mistake {mk} ===\n")
        print(f"Overall: {ratio_str(mistake_counts[mk], overall_denom)}")
        for p in [1, 2]:
            if mk in (4, 5):
                denom = num_games
            else:
                denom = moves_by_player[p]
            print(f"Player {p}: {ratio_str(mistake_counts_by_player[p][mk], denom)}")
        for r in [1, 2, 3]:
            if mk in (4, 5):
                denom = num_games
            else:
                denom = moves_by_round[r]
            mk_count = mistake_counts_by_round[r].get(mk, 0)
            print(f"Round {r}: {ratio_str(mk_count, denom)}")
        print()

    for mk in [1, 2, 3, 4, 5]:
        print_mistake_results(mk)
    return mistake_counts

def plot_mistakes_radar_multiple_circles(circles_mistake_counts, model_name):
    """
    Plots a single radar chart with multiple "circles" (series),
    each circle represented by a distinct color.

    circles_mistake_counts: dict where:
       - key: circle name (str), e.g. "circle_0"
       - value: dict of {mistake_id: count}, e.g., {1:10, 2:4, 3:5, 4:3, 5:2}
    """

    labels = [1, 2, 3, 4, 5]
    n_labels = len(labels)

    angles = np.linspace(0, 2 * np.pi, n_labels, endpoint=False).tolist()
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    colors = plt.cm.tab10(np.linspace(0, 1, len(circles_mistake_counts)))

    for (circle_name, mistake_counts), color in zip(circles_mistake_counts.items(), colors):
        values = [mistake_counts.get(mk, 0) for mk in labels]
        values += [values[0]]

        ax.plot(angles, values, 'o-', linewidth=2, label=circle_name, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), [f"Dominated Strategy {k}" for k in labels])
    ax.set_title(f"Dominated Strategies Count for {model_name} by Circle", y=1.1)
    ax.grid(True)

  
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.show()

'''
- Mistake 1: Making an offer worse than your previous offer. This occurs when you make an offer that is worse than the offer that is currently on the table ie if you make an offer that is worse than the offer that is currently on the table for you to accept. 
- Mistake 2: Making an offer worse for you than your outside offer. This happens if you propose giving away so much that what you keep is worth less than your guaranteed alternative, which is your outside offer.
- Mistake 3: Offering no items or all items. Offering nothing (or everything) to the opponent (in the early or middle rounds) can be a clear suboptimal move. 
- Mistake 4: Accepting an offer worse for you than your outside offer. This occurs if you accept a division that yields a payoff lower than your guaranteed fallback.
- Mistake 5: Walking away from an offer better than your outside offer. This occurs when you reject a division that actually yields a higher payoff than your fallback.
'''
circles = [0, 1, 2, 3, 4, 5, 6]

ds_data_4o = {}
ds_data_gemini = {}
for circle in circles:
    print(f"Analyzing circle {circle}")
    print("="*100)
    #json_FILE_PATH = f"/Users/gabesmithline/Desktop/caif_negotiation/experiments/4o_final/4o_1_28_2025_100_circle_{circle}.json"
    #json_FILE_PATH = f"/Users/gabesmithline/Desktop/caif_negotiation/experiments/gemini_2.0_final/gemini_2.0_1_28_2025_circle_{circle}.json"
    #json_FILE_PATH = f"/Users/gabesmithline/Desktop/caif_negotiation/experiments/o3-mini/all_game_data_2_4_2025_100_o3-mini_circle_{circle}.json"
    #json_FILE_PATH = f"/Users/gabesmithline/Desktop/caif_negotiation/experiments/4o_300/4o_2_9_2025_300_circle_{circle}.json"
    #json_FILE_PATH = f"/Users/gabesmithline/Desktop/caif_negotiation/experiments/4o_small_final/4o_2_6_2025_100_circle_{circle}_small.json"
    #json_FILE_PATH = f"/Users/gabesmithline/Desktop/caif_negotiation/experiments/cross_play/4o_vs_o3/all_game_data_2_9_2025_100_4o_vs_o3_mini_circle_{circle}.json"
    json_FILE_PATH = f"/Users/gabesmithline/Desktop/caif_negotiation/experiments/claude_3.7/all_game_data_2_24_2025_100_anthropic_3.7_sonnet_vs_anthropic_3.7_sonnet_circle_p1_{circle}_circle_p2_{circle}.json"
    #json_FILE_PATH = f"/Users/gabesmithline/Desktop/caif_negotiation/metrics/test_mistakes_5games.json"
    mistake_counts = analyze_circle6(json_FILE_PATH)
    print(mistake_counts)
    ds_data_gemini[f"sonnet 3.7_circle_{circle}"] = mistake_counts

plot_mistakes_radar_multiple_circles(ds_data_gemini, "sonnet 3.7")