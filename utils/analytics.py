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
    #check type in valuations & allocations
    if valuations is None or allocation is None:
        print("HERE")
    elif any(x is None for x in valuations) or any(x is None for x in allocation):
        print("HERE2")
    return sum(v * a for v, a in zip(valuations, allocation))

def detect_mistakes(current_move_info, previous_move_info_player):
    mistakes_found = []
    action = current_move_info["action"]
    my_val_cur = current_move_info["my_value_current_offer"]
    my_val_prev = current_move_info["my_value_previous_own_offer"]
    out_offer = current_move_info["my_outside_offer"]
    allocation_to_opp = current_move_info["allocation_offered_to_opponent"]
    allocation_i_keep = current_move_info["allocation_i_keep"]
    is_game_ending = current_move_info["is_game_ending"]

    if action == "COUNTEROFFER" and not is_game_ending:
        if my_val_prev is not None and my_val_prev > my_val_cur:
            mistakes_found.append(1)
        if my_val_cur < out_offer:
            mistakes_found.append(2)
        if allocation_to_opp is not None and allocation_i_keep is not None:
            sum_offered = sum(allocation_to_opp)
            sum_kept = sum(allocation_i_keep)
            total_items = sum_offered + sum_kept
            if total_items > 0:
                if sum_offered == 0 or sum_offered == total_items:
                    mistakes_found.append(3)

    if action == "ACCEPT" and is_game_ending:
        if my_val_cur < out_offer:
            mistakes_found.append(4)

    if action == "WALK" and is_game_ending:
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
        total_rounds = 3
        current_round = 1
        i = 0

        # We iterate through at most total_rounds, each round can have up to two moves
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
                # If something is off, skip this round
                current_round += 1
                continue

            try:
                parsed_info = parse_valuations_from_prompt(prompt)
            except ValueError:
                current_round += 1
                continue
            print(parsed_info)
            print(parsed_info["player_id"])
            player_id = parsed_info["player_id"]
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
                # Action is something we don't track, do not increment round multiple times
                continue

            my_value_previous_own_offer = None
            if last_move_info[player_id] and last_move_info[player_id]["action"] == "COUNTEROFFER":
                my_value_previous_own_offer = last_move_info[player_id]["my_value_current_offer"]

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
                last_move_info.get(player_id)
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
                last_move_info[player_id] = {
                    "action": action,
                    "my_value_current_offer": current_value_for_self,
                    "allocation_offered_to_opponent": allocation_offered_to_opponent,
                    "allocation_i_keep": allocation_i_keep
                }
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
                        
                        current_value_for_self2 = compute_allocation_value(valuations2, our_alloc2) * discount_factor2
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
                        #if our_alloc2 is None or valuations2 is None:
                           #print(prompt)
                            #print(response)
                            #print(action)
                        
                        current_value_for_self2 = compute_allocation_value(valuations2, our_alloc2) * discount_factor2
                        allocation_offered_to_opponent2 = None
                        allocation_i_keep2 = our_alloc2
                    else:
                        current_value_for_self2 = 0.0
                        allocation_offered_to_opponent2 = None
                        allocation_i_keep2 = None
                else:
                    # Unknown action, do not truncate the round count
                    current_round += 1
                    continue

                my_value_previous_own_offer2 = None
                if last_move_info[player_id2] and last_move_info[player_id2]["action"] == "COUNTEROFFER":
                    my_value_previous_own_offer2 = last_move_info[player_id2]["my_value_current_offer"]

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
                    last_move_info.get(player_id2)
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
                if is_game_ending2:
                    current_round += 1
                    break

            # End of this round if the second move didn't end the game
            else:
                # If there's no second move, we still consider the round done
                pass

            # Move to the next round
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

    # For simplicity, we'll assume mistakes are always 1..5
    labels = [1, 2, 3, 4, 5]
    n_labels = len(labels)

    # Create angles for the radar plot
    angles = np.linspace(0, 2 * np.pi, n_labels, endpoint=False).tolist()
    # Close the circle
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    # Use a color map to assign a different color to each circle
    colors = plt.cm.tab10(np.linspace(0, 1, len(circles_mistake_counts)))

    for (circle_name, mistake_counts), color in zip(circles_mistake_counts.items(), colors):
        # Extract values in the sorted mistake order
        # (so that label index 0 => mistake #1, etc.)
        values = [mistake_counts.get(mk, 0) for mk in labels]
        values += [values[0]]  # close the polygon

        # Plot the line for this circle
        ax.plot(angles, values, 'o-', linewidth=2, label=circle_name, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

    # Set up the axis ticks and labels
    ax.set_thetagrids(np.degrees(angles[:-1]), [f"Dominated Strategy {k}" for k in labels])
    ax.set_title(f"Dominated Strategies for {model_name} by Circle", y=1.1)
    ax.grid(True)

  
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.show()



import json

def parse_player_id(prompt_text):
    """
    Identifies whether prompt says "You are Player 1" or "You are Player 2".
    Returns int(1) or int(2).
    Raises ValueError if not found.
    """
    if "You are Player 1" in prompt_text:
        return 1
    elif "You are Player 2" in prompt_text:
        return 2
    raise ValueError("Could not determine player from prompt.")

def count_game_actions(json_file_path):
    """
    Similar to analyze_circle6, but counts actions:
      - ACCEPT
      - WALK
      - INVALID WALK
      - COUNTEROFFER
    by round (1..3), by player (1|2), and a total.
    
    Returns:
      stats, num_games

    stats is a dict like:
      {
        "ACCEPT": {
          "rounds": {1: int, 2: int, 3: int},
          "players": {1: int, 2: int},
          "total": int
        },
        "WALK": {...}, 
        "INVALID WALK": {...}, 
        "COUNTEROFFER": {...}
      }

    num_games is the number of games in that JSON file (len(all_game_data)).
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_game_data = data["all_game_data"]
    num_games = len(all_game_data)

    actions_of_interest = ["ACCEPT", "WALK", "INVALID WALK", "COUNTEROFFER"]
    # Initialize counters
    stats = {
        action: {
            "rounds": {1: 0, 2: 0, 3: 0},
            "players": {1: 0, 2: 0},
            "total": 0
        }
        for action in actions_of_interest
    }

    for game in all_game_data:
        round_data = game["round_data"]
        current_round = 1
        total_rounds = 3
        i = 0

        while i < len(round_data) and current_round <= total_rounds:
            # First move
            move1 = round_data[i]
            i += 1

            action1 = move1.get("action", None)
            prompt1 = move1.get("prompt", "")
            try:
                p1 = parse_player_id(prompt1)
            except ValueError:
                current_round += 1
                continue

            if action1 in stats:
                stats[action1]["rounds"][current_round] += 1
                stats[action1]["players"][p1] += 1
                stats[action1]["total"] += 1

            # End round if terminating action
            if action1 in ["ACCEPT", "WALK", "INVALID WALK"]:
                current_round += 1
                continue

            # Second move
            if i < len(round_data):
                move2 = round_data[i]
                i += 1

                action2 = move2.get("action", None)
                prompt2 = move2.get("prompt", "")
                try:
                    p2 = parse_player_id(prompt2)
                except ValueError:
                    current_round += 1
                    continue

                if action2 in stats:
                    stats[action2]["rounds"][current_round] += 1
                    stats[action2]["players"][p2] += 1
                    stats[action2]["total"] += 1

                if action2 in ["ACCEPT", "WALK", "INVALID WALK"]:
                    current_round += 1
                    continue

            # If no terminating action, just move to the next round
            current_round += 1

    return stats, num_games


def print_game_action_stats(stats):
    """
    Pretty-prints the raw counts in a table:
      Rows = Action (#accepts, #walks, #invalid walks, #counter offers)
      Columns = Round 1, Round 2, Round 3, Total, Player 1, Player 2
    """
    header_row = (
        f"{'Game Stats:':<16}"
        f"{'Round 1':>8}"
        f"{'Round 2':>8}"
        f"{'Round 3':>8}"
        f"{'Total':>8}"
        f"{'P1':>8}"
        f"{'P2':>8}"
    )
    print(header_row)
    print("-" * len(header_row))

    action_labels = {
        "ACCEPT": "#accepts",
        "WALK": "#walks",
        "INVALID WALK": "#invalid walks",
        "COUNTEROFFER": "#counter offers"
    }

    for action in ["ACCEPT", "WALK", "INVALID WALK", "COUNTEROFFER"]:
        label = action_labels[action]
        r1 = stats[action]["rounds"][1]
        r2 = stats[action]["rounds"][2]
        r3 = stats[action]["rounds"][3]
        tot = stats[action]["total"]
        p1 = stats[action]["players"][1]
        p2 = stats[action]["players"][2]

        row_str = (
            f"{label:<16}"
            f"{r1:>8}"
            f"{r2:>8}"
            f"{r3:>8}"
            f"{tot:>8}"
            f"{p1:>8}"
            f"{p2:>8}"
        )
        print(row_str)

def aggregate_stats_across_circles(stats_list_with_gamecounts):
    """
    Takes a list of (stats, num_games) from multiple circles
    and aggregates them. We sum up all counts across circles,
    and sum up the total number of games.

    Returns:
      (aggregated_stats, overall_num_games)
    """

    actions_of_interest = ["ACCEPT", "WALK", "INVALID WALK", "COUNTEROFFER"]
    # Initialize an empty aggregator
    agg_stats = {
        action: {
            "rounds": {1: 0, 2: 0, 3: 0},
            "players": {1: 0, 2: 0},
            "total": 0
        }
        for action in actions_of_interest
    }

    total_games = 0

    # Sum them up
    for (stats, num_games) in stats_list_with_gamecounts:
        total_games += num_games
        for action in actions_of_interest:
            agg_stats[action]["total"] += stats[action]["total"]
            for r in [1,2,3]:
                agg_stats[action]["rounds"][r] += stats[action]["rounds"][r]
            for p in [1,2]:
                agg_stats[action]["players"][p] += stats[action]["players"][p]

    return agg_stats, total_games


def print_averages_across_all_games(agg_stats, total_games):
    """
    Prints a table similar to print_game_action_stats,
    but each cell is an average per game = (counts / total_games).
    """

    if total_games == 0:
        print("No games to compute averages.")
        return

    header_row = (
        f"{'Avg Stats:':<16}"
        f"{'Round1':>10}"
        f"{'Round2':>10}"
        f"{'Round3':>10}"
        f"{'Total':>10}"
        f"{'P1':>10}"
        f"{'P2':>10}"
    )
    print(header_row)
    print("-" * len(header_row))

    action_labels = {
        "ACCEPT": "#accepts",
        "WALK": "#walks",
        "INVALID WALK": "#invalid walks",
        "COUNTEROFFER": "#counter offers"
    }

    for action in ["ACCEPT", "WALK", "INVALID WALK", "COUNTEROFFER"]:
        label = action_labels[action]
        r1 = agg_stats[action]["rounds"][1] / total_games
        r2 = agg_stats[action]["rounds"][2] / total_games
        r3 = agg_stats[action]["rounds"][3] / total_games
        tot = agg_stats[action]["total"] / total_games
        p1 = agg_stats[action]["players"][1] / total_games
        p2 = agg_stats[action]["players"][2] / total_games

        row_str = (
            f"{label:<16}"
            f"{r1:>10.2f}"
            f"{r2:>10.2f}"
            f"{r3:>10.2f}"
            f"{tot:>10.2f}"
            f"{p1:>10.2f}"
            f"{p2:>10.2f}"
        )
        print(row_str)


# -----------------------------------------------------------------------------------
# Example usage in a separate file or a notebook:


import json
import math
import re
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Helper functions (Parsing, computations, mistake checks)
##############################################################################

def parse_player_id(prompt_text):
    """
    Reads the prompt text to see if it says 'You are Player 1' or 'You are Player 2'.
    Returns 1 or 2. Raises ValueError if neither found.
    """
    if "You are Player 1" in prompt_text:
        return 1
    elif "You are Player 2" in prompt_text:
        return 2
    else:
        raise ValueError("Could not determine player from prompt.")

def parse_valuations_from_prompt_private_values(prompt_text):
    """
    Attempts to parse lines like:
       'Your private values are A for item 1, B for item 2, C for item 3, D for item 4, E for item 5.'
    Returns a list of floats [A,B,C,D,E] if found, else None.
    """
    pattern_vals = re.compile(
        r"Your private values are\s+(\d+)\s+for item 1,\s*(\d+)\s+for item 2,\s*(\d+)\s+for item 3,\s*(\d+)\s+for item 4,\s*(\d+)\s+for item 5\."
    )
    match_vals = pattern_vals.search(prompt_text)
    if match_vals:
        return [float(x) for x in match_vals.groups()]
    return None

def parse_outside_offer_from_prompt(prompt_text):
    """
    Parses a line like: 
       'Your outside offer value is Y.' 
    Returns float(Y) or None if not found.
    """
    pattern_out = re.compile(r"Your outside offer value is\s+(\d+(\.\d+)?)\.")
    match_out = pattern_out.search(prompt_text)
    if match_out:
        return float(match_out.group(1))
    return None

def parse_items_from_prompt(prompt_text):
    """
    Parses a line like:
       'There are 1 units of item 1, 0 units of item 2, 4 units of item 3, 3 units of item 4, 2 units of item 5 to divide.'
    Returns a list of ints like [1, 0, 4, 3, 2] or None if not found.
    """
    pattern_items = re.compile(
        r"There are\s+(\d+)\s+units of item 1,\s+(\d+)\s+units of item 2,\s+(\d+)\s+units of item 3,\s+(\d+)\s+units of item 4,\s+(\d+)\s+units of item 5 to divide\."
    )
    match_items = pattern_items.search(prompt_text)
    if match_items:
        return [int(x) for x in match_items.groups()]
    return None

def parse_offer_from_response(response_text):
    """
    Pulls anything like: "offer": [x1, x2, x3, x4, x5] out of the JSON in the response.
    Returns a list of floats or None if not found.
    """
    pattern = re.compile(
        r"\"offer\"\s*:\s*\[\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*\]"
    )
    match = pattern.search(response_text)
    if match:
        return [float(g) for g in match.groups()]
    return None

def compute_dotproduct(offer, valuations):
    """Returns sum(offer[i] * valuations[i]) for i in [0..4]."""
    return sum(o * v for o, v in zip(offer, valuations))

def check_5_mistakes(current_action,
                     my_val_cur,
                     my_val_prev,
                     out_offer,
                     new_offer,
                     total_items):
    """
    Classic 5 mistakes/dominated actions:
      1) Offering worse than your *previous* offer.
      2) Offering worse than your outside offer (COUNTEROFFER only).
      3) Offering no items or all items (COUNTEROFFER only).
      4) ACCEPT something worse than outside.
      5) WALK away from something better than outside.
    Returns a dict {1: bool, 2: bool, 3: bool, 4: bool, 5: bool}.
    """
    mistakes = {k: False for k in range(1, 6)}

    def approx_lt(a, b):
        return (a is not None) and (b is not None) and (round(a, 6) < round(b, 6))

    if current_action == "COUNTEROFFER":
        # (3) Offering no items or all items
        if new_offer is not None:
            if all(q == 0 for q in new_offer) or all(q == t for q, t in zip(new_offer, total_items)):
                mistakes[3] = True

        # (1) Current offer is worse for you than previous
        if my_val_cur is not None and my_val_prev is not None:
            if approx_lt(my_val_cur, my_val_prev):
                mistakes[1] = True

        # (2) Offer worse than outside
        if my_val_cur is not None and out_offer is not None:
            if approx_lt(my_val_cur, out_offer):
                mistakes[2] = True

    elif current_action == "ACCEPT":
        # (4) Accept something worse than outside
        if my_val_cur is not None and out_offer is not None:
            if approx_lt(my_val_cur, out_offer):
                mistakes[4] = True

    elif current_action == "WALK":
        # (5) Walk away from something better than outside
        if my_val_cur is not None and out_offer is not None:
            if my_val_cur > out_offer:
                mistakes[5] = True

    return mistakes

def compute_gini(player_1_val, player_2_val):
    """
    For two players, a typical formula is: G = |x1 - x2| / (x1 + x2).
    If the sum is zero or negative, we treat G = 0 to avoid division by zero.
    """
    total = player_1_val + player_2_val
    if total <= 0:
        return 0.0
    return abs(player_1_val - player_2_val) / (4 * total)

def compute_nash(player_1_val, player_2_val):
    """
    For 2 players: Nash = sqrt(max(0, x1) * max(0, x2)).
    """
    x1 = max(0.0, player_1_val)
    x2 = max(0.0, player_2_val)
    return math.sqrt(x1 * x2)

def compute_rawls(player_1_val, player_2_val):
    """Rawls = min(x1, x2)."""
    return min(player_1_val, player_2_val)

def compute_utilitarian(player_1_val, player_2_val):
    """Utilitarian = x1 + x2."""
    return player_1_val + player_2_val

def compute_jain_fairness(player_1_val, player_2_val):
    """
    Jain's fairness index for 2 players: (x1 + x2)^2 / (2*(x1^2 + x2^2)).
    """
    x1 = max(0, player_1_val)
    x2 = max(0, player_2_val)
    s = x1 + x2
    q = x1*x1 + x2*x2
    if q == 0:
        return 0.0
    return (s * s) / (2.0 * q)

def compute_mean_absolute_diff(player_1_val, player_2_val):
    """
    Mean absolute difference for n=2 is |x1 - x2| / 2.
    """
    return abs(player_1_val - player_2_val) / 2.0

def compute_security_regret(player_val, outside_val):
    """
    Security/regret level: max(0, outside_val - player_val).
    If you get less than outside, it's positive; if you get more, it's 0.
    """
    return max(0.0, outside_val - player_val)


def compute_and_plot_metrics(json_file_path, circle_name="circle_0", llm_name="unknown_llm", gamma=0.9):
    """
    Reads JSON data for a single circle containing multiple games,
    computes relevant metrics at each step, and plots them by step_index.

    Key points to address:
      • If a final step is WALK or INVALID WALK, player 1's final payoff is p1_outside,
        and player 2's final payoff is p2_outside.
      • If a final step is ACCEPT, we use the most recent COUNTEROFFER's allocation as
        the final accepted offer. (We track last_counter_offer as we parse.)
      • Fairness metrics (Nash, Gini, etc.) should be computed at every step where
        there is a definable (p1_val, p2_val), not forced to zero if, in fact,
        we do have an offer at that step.
      • The x-axis of the graphs must be accurate. If the game never reaches
        a certain half-step (e.g. 3.5), then we should not see it on the chart.

    We adopt the convention:
      - When Player 1 acts in round N, step_index = float(N).
      - When Player 2 acts in round N, step_index = float(N) + 0.5,
        then we increment round_num to N+1 if the game continues.

    That way, if the game ends after Player 1's action in round 3 (no Player 2 in round 3),
    we do NOT see 3.5 in the data.
    """

    import json
    import math
    import re
    import numpy as np
    import matplotlib.pyplot as plt

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_game_data = data.get("all_game_data", [])
    step_metrics = []  # collect one record per step across all games

    
    def parse_player_id(prompt_text):
        if "You are Player 1" in prompt_text:
            return 1
        elif "You are Player 2" in prompt_text:
            return 2
        else:
            raise ValueError("Could not determine player from prompt.")

    def parse_valuations_from_prompt(prompt_text):
        pattern_vals = re.compile(
            r"Your private values are\s+(\d+)\s+for item 1,\s*(\d+)\s+for item 2,\s*(\d+)\s+for item 3,\s*(\d+)\s+for item 4,\s*(\d+)\s+for item 5\."
        )
        match_vals = pattern_vals.search(prompt_text)
        if match_vals:
            return [float(x) for x in match_vals.groups()]
        return None

    def parse_outside_offer_from_prompt(prompt_text):
        pattern_out = re.compile(r"Your outside offer value is\s+(\d+(\.\d+)?)\.")
        match_out = pattern_out.search(prompt_text)
        if match_out:
            return float(match_out.group(1))
        return None

    def parse_items_from_prompt(prompt_text):
        pattern_items = re.compile(
            r"There are\s+(\d+)\s+units of item 1,\s+(\d+)\s+units of item 2,\s+(\d+)\s+units of item 3,\s+(\d+)\s+units of item 4,\s+(\d+)\s+units of item 5 to divide\."
        )
        match_items = pattern_items.search(prompt_text)
        if match_items:
            return [int(x) for x in match_items.groups()]
        return None

    def parse_offer_from_response(response_text):
        pattern = re.compile(
            r"\"offer\"\s*:\s*\[\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*\]"
        )
        match = pattern.search(response_text)
        if match:
            return [float(g) for g in match.groups()]
        return None

    def compute_dotproduct(offer, valuations):
        return sum(o * v for o, v in zip(offer, valuations))

    def compute_gini(player_1_val, player_2_val):
        total = player_1_val + player_2_val
        if total <= 0:
            return 0.0
        return abs(player_1_val - player_2_val) / (4 * total)

    def compute_nash(player_1_val, player_2_val):
        x1 = max(0.0, player_1_val)
        x2 = max(0.0, player_2_val)
        return math.sqrt(x1 * x2)

    def compute_rawls(player_1_val, player_2_val):
        return min(player_1_val, player_2_val)

    def compute_utilitarian(player_1_val, player_2_val):
        return player_1_val + player_2_val

    def compute_jain_fairness(player_1_val, player_2_val):
        x1 = max(0.0, player_1_val)
        x2 = max(0.0, player_2_val)
        s = x1 + x2
        q = x1*x1 + x2*x2
        if q == 0:
            return 0.0
        return (s*s)/(2.0*q)

    def compute_mean_absolute_diff(player_1_val, player_2_val):
        return abs(player_1_val - player_2_val) / 2.0

    def compute_security_regret(player_val, outside_val):
        return max(0.0, outside_val - player_val)

    def check_5_mistakes(current_action,
                         my_val_cur,
                         my_val_prev,
                         out_offer,
                         new_offer,
                         total_items):
        mistakes = {k: False for k in range(1, 6)}
        def approx_lt(a, b): 
            return (a is not None) and (b is not None) and (round(a, 6) < round(b, 6))

        if current_action == "COUNTEROFFER":
            if new_offer is not None and total_items is not None:
                # (3) Offering no items or all items
                if all(q == 0 for q in new_offer) or all(q == t for q, t in zip(new_offer, total_items)):
                    mistakes[3] = True
            # (1) Offer is worse for you than your previous
            if my_val_cur is not None and my_val_prev is not None:
                if approx_lt(my_val_cur, my_val_prev):
                    mistakes[1] = True
            # (2) Offer is worse than your outside
            if my_val_cur is not None and out_offer is not None:
                if approx_lt(my_val_cur, out_offer):
                    mistakes[2] = True

        elif current_action == "ACCEPT":
            # (4) Accept something worse than outside
            if my_val_cur is not None and out_offer is not None:
                if approx_lt(my_val_cur, out_offer):
                    mistakes[4] = True

        elif current_action == "WALK":
            # (5) Walk away from something better than outside
            if my_val_cur is not None and out_offer is not None:
                if my_val_cur > out_offer:
                    mistakes[5] = True
        return mistakes

    ####################################################################
    # PARSE GAMES, TRACK STEP INDICES, ETC.
    ####################################################################

    for game_idx, game_info in enumerate(all_game_data):
        round_data = game_info.get("round_data", [])

        p1_valuations, p2_valuations = None, None
        p1_outside, p2_outside = None, None
        total_items = None

        # Gather environment info (valuations, outside offers, items)
        for move in round_data:
            ptxt = move.get("prompt", "")
            try:
                pl = parse_player_id(ptxt)
            except ValueError:
                pl = None
            v = parse_valuations_from_prompt(ptxt)
            out_val = parse_outside_offer_from_prompt(ptxt)
            items_found = parse_items_from_prompt(ptxt)
            if items_found:
                total_items = items_found
            if pl == 1:
                if v is not None:
                    p1_valuations = v
                if out_val is not None:
                    p1_outside = out_val
            elif pl == 2:
                if v is not None:
                    p2_valuations = v
                if out_val is not None:
                    p2_outside = out_val

        last_counter_offer_val_p1 = None
        last_counter_offer_val_p2 = None

        prev_p1_val, prev_p2_val = None, None

        current_round_num = 1
        i = 0
        while i < len(round_data):
            move = round_data[i]
            i += 1
            ptxt = move.get("prompt", "")
            rtxt = move.get("response", "")
            action = move.get("action", "")
            try:
                current_player = parse_player_id(ptxt)
            except ValueError:
                current_player = None

            new_offer = parse_offer_from_response(rtxt)
            # Step index logic:
            #  - If it's player 1's turn => step = float(current_round_num)
            #  - If it's player 2's turn => step = float(current_round_num) + 0.5, then we'll increment round_num
            if current_player == 1:
                step_index = float(current_round_num)
            elif current_player == 2:
                step_index = float(current_round_num) + 0.5
            else:
                # fallback
                step_index = float(current_round_num)

            # Evaluate the new offer's valuations if present
            p1_offer_val = None
            p2_offer_val = None
            if p1_valuations and p2_valuations and new_offer and len(new_offer) == len(p1_valuations):
                p1_offer_val = compute_dotproduct(new_offer, p1_valuations)
                p2_offer_val = compute_dotproduct(new_offer, p2_valuations)

            if action == "COUNTEROFFER":
                if current_player == 1:
                    last_counter_offer_val_p1 = p1_offer_val
                    last_counter_offer_val_p2 = p2_offer_val
                elif current_player == 2:
                    last_counter_offer_val_p1 = p1_offer_val
                    last_counter_offer_val_p2 = p2_offer_val

            if current_player == 1:
                my_val_cur = p1_offer_val
                my_val_prev = prev_p1_val
                out_val = p1_outside
            elif current_player == 2:
                my_val_cur = p2_offer_val
                my_val_prev = prev_p2_val
                out_val = p2_outside
            else:
                my_val_cur = None
                my_val_prev = None
                out_val = None

            mistakes_dict = check_5_mistakes(
                current_action=action,
                my_val_cur=my_val_cur,
                my_val_prev=my_val_prev,
                out_offer=out_val,
                new_offer=new_offer,
                total_items=total_items or []
            )

            # Update previous val for whoever made a COUNTEROFFER
            if action == "COUNTEROFFER":
                if current_player == 1 and p1_offer_val is not None:
                    prev_p1_val = p1_offer_val
                elif current_player == 2 and p2_offer_val is not None:
                    prev_p2_val = p2_offer_val

            # Figure out final (p1_val, p2_val) for fairness metrics at this step
            p1_val_final, p2_val_final = None, None
            if action == "COUNTEROFFER":
                # We have a definable offer from current_player
                if p1_offer_val is not None and p2_offer_val is not None:
                    p1_val_final = p1_offer_val
                    p2_val_final = p2_offer_val
            elif action == "ACCEPT":
                # Use the last counter
                if last_counter_offer_val_p1 is not None and last_counter_offer_val_p2 is not None:
                    p1_val_final = last_counter_offer_val_p1
                    p2_val_final = last_counter_offer_val_p2
            elif action in ["WALK", "INVALID WALK"]:
                # Both get outside
                if (p1_outside is not None) and (p2_outside is not None):
                    p1_val_final = p1_outside
                    p2_val_final = p2_outside

            # Compute fairness metrics (undiscounted) if p1_val_final/p2_val_final
            if (p1_val_final is not None) and (p2_val_final is not None):
                gini_val = compute_gini(p1_val_final, p2_val_final)
                nash_val = compute_nash(p1_val_final, p2_val_final)
                rawls_val = compute_rawls(p1_val_final, p2_val_final)
                util_val = compute_utilitarian(p1_val_final, p2_val_final)
                jain_val = compute_jain_fairness(p1_val_final, p2_val_final)
                mad_val = compute_mean_absolute_diff(p1_val_final, p2_val_final)

                # For security_regret, we might consider only the current player's perspective.
                sec_reg = None
                if current_player == 1 and p1_outside is not None:
                    sec_reg = compute_security_regret(p1_val_final, p1_outside)
                elif current_player == 2 and p2_outside is not None:
                    sec_reg = compute_security_regret(p2_val_final, p2_outside)
            else:
                gini_val, nash_val, rawls_val = None, None, None
                util_val, jain_val, mad_val, sec_reg = None, None, None, None

            # Concession size if this is a COUNTEROFFER
            concession_size = None
            if (action == "COUNTEROFFER") and (my_val_cur is not None) and (my_val_prev is not None):
                diff_val = my_val_prev - my_val_cur
                if diff_val > 0:
                    concession_size = diff_val

            step_metrics.append({
                "circle_name": circle_name,
                "llm_name": llm_name,
                "game_index": game_idx,
                "round_number": current_round_num,
                "player": current_player,
                "step_index": step_index,
                "action": action,

                "p1_undiscounted_value": p1_val_final,
                "p2_undiscounted_value": p2_val_final,

                "gini_coefficient": gini_val,
                "nash_welfare": nash_val,
                "rawlsian_welfare": rawls_val,
                "utilitarian_welfare": util_val,
                "jain_fairness_index": jain_val,
                "mean_absolute_difference": mad_val,
                "security_regret": sec_reg,
                "concession_size": concession_size,

                "mistake_1": mistakes_dict[1],
                "mistake_2": mistakes_dict[2],
                "mistake_3": mistakes_dict[3],
                "mistake_4": mistakes_dict[4],
                "mistake_5": mistakes_dict[5],
            })

            # Only now do we increment round if it's player 2's move
            if current_player == 2:
                current_round_num += 1

    ########################################################################
    # Build line charts for p1/p2 undiscounted values. 
    # x-axis = step_index. If there's truly no step 3.5 in your data,
    # we won't see it on the chart.
    ########################################################################
    circle_data = [r for r in step_metrics if r["circle_name"] == circle_name]
    all_steps = sorted(set(rec["step_index"] for rec in circle_data))

    # We'll compute an average p1/p2 value at each step, 
    # and figure out "own" vs. "other" marker by majority player in that step.
    step_lookup = {}
    for rec in circle_data:
        sidx = rec["step_index"]
        if sidx not in step_lookup:
            step_lookup[sidx] = []
        step_lookup[sidx].append(rec)

    p1_own_x, p1_own_y = [], []
    p1_other_x, p1_other_y = [], []
    p2_own_x, p2_own_y = [], []
    p2_other_x, p2_other_y = [], []

    for s in all_steps:
        these_recs = step_lookup[s]
        p1_vals = [x["p1_undiscounted_value"] for x in these_recs if x["p1_undiscounted_value"] is not None]
        p2_vals = [x["p2_undiscounted_value"] for x in these_recs if x["p2_undiscounted_value"] is not None]
        p1_avg = sum(p1_vals)/len(p1_vals) if p1_vals else None
        p2_avg = sum(p2_vals)/len(p2_vals) if p2_vals else None

        # majority player at this step => "own" move, the other => "other" move
        players_in_step = [x["player"] for x in these_recs if x["player"] in [1,2]]
        if not players_in_step:
            continue
        p1_count = players_in_step.count(1)
        p2_count = players_in_step.count(2)

        if p1_count >= p2_count:  # treat as P1's move
            if p1_avg is not None:
                p1_own_x.append(s)
                p1_own_y.append(p1_avg)
            if p2_avg is not None:
                p2_other_x.append(s)
                p2_other_y.append(p2_avg)
        else:  # P2's move
            if p2_avg is not None:
                p2_own_x.append(s)
                p2_own_y.append(p2_avg)
            if p1_avg is not None:
                p1_other_x.append(s)
                p1_other_y.append(p1_avg)

    # Build line for P1
    p1_combined_steps = p1_own_x + p1_other_x
    p1_combined_vals  = p1_own_y + p1_other_y
    p1_pairs_sorted = sorted(zip(p1_combined_steps, p1_combined_vals), key=lambda x: x[0])
    p1_line_x = [z[0] for z in p1_pairs_sorted]
    p1_line_y = [z[1] for z in p1_pairs_sorted]

    # Build line for P2
    p2_combined_steps = p2_own_x + p2_other_x
    p2_combined_vals  = p2_own_y + p2_other_y
    p2_pairs_sorted = sorted(zip(p2_combined_steps, p2_combined_vals), key=lambda x: x[0])
    p2_line_x = [z[0] for z in p2_pairs_sorted]
    p2_line_y = [z[1] for z in p2_pairs_sorted]

    plt.figure(figsize=(9,5))
    # P1 line
    plt.plot(p1_line_x, p1_line_y, 'b-', label="P1 average value")
    plt.scatter(p1_own_x,   p1_own_y,   marker='o', color='blue', label="P1 own move")
    plt.scatter(p1_other_x, p1_other_y, marker='s', color='blue', label="P1 other move")

    # P2 line
    plt.plot(p2_line_x, p2_line_y, 'r-', label="P2 average value")
    plt.scatter(p2_own_x,   p2_own_y,   marker='o', color='red', label="P2 own move")
    plt.scatter(p2_other_x, p2_other_y, marker='s', color='red', label="P2 other move")

    plt.title(f"Avg Undiscounted Value by Step (Circle: {circle_name}, LLM: {llm_name})")
    plt.xlabel("Step index (1, 1.5, 2, 2.5, ...)")
    plt.ylabel("Undiscounted Value")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(all_steps, [str(a) for a in all_steps])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    ########################################################################
    # Plot fairness metrics across steps that actually have p1_val and p2_val.
    ########################################################################
    metrics_to_plot = [
        "gini_coefficient",
        "nash_welfare",
        "rawlsian_welfare",
        "utilitarian_welfare",
        "jain_fairness_index",
        "mean_absolute_difference",
        "security_regret",
        "concession_size"
    ]

    step_lookup_metrics = {}
    for rec in circle_data:
        sidx = rec["step_index"]
        if sidx not in step_lookup_metrics:
            step_lookup_metrics[sidx] = []
        step_lookup_metrics[sidx].append(rec)

    sorted_steps = sorted(step_lookup_metrics.keys())

    for metric in metrics_to_plot:
        x_vals, y_vals = [], []
        for s in sorted_steps:
            relevant = [r[metric] for r in step_lookup_metrics[s] if r[metric] is not None]
            if relevant:
                avg_val = sum(relevant)/len(relevant)
                x_vals.append(s)
                y_vals.append(avg_val)
        if not x_vals:
            continue

        plt.figure(figsize=(7,4))
        plt.plot(x_vals, y_vals, 'bo-', alpha=0.8)
        plt.title(f"Avg {metric} by Step (Circle: {circle_name}, LLM: {llm_name})")
        plt.xlabel("Step index")
        plt.ylabel(metric)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(x_vals, [str(s) for s in x_vals])
        plt.tight_layout()
        plt.show()
    
    ########################################################################
    # Bar chart for each metric aggregated by LLM. 
    # (If multiple LLMs exist in a single file, it will show a bar for each LLM).
    ########################################################################
    llm_map = {}
    for rec in step_metrics:
        rec_llm = rec["llm_name"]
        if rec_llm not in llm_map:
            llm_map[rec_llm] = []
        llm_map[rec_llm].append(rec)

    for metric in metrics_to_plot:
        labels = []
        vals_list = []
        for cur_llm in sorted(llm_map.keys()):
            relevant_vals = [rr[metric] for rr in llm_map[cur_llm] if rr[metric] is not None]
            if relevant_vals:
                mean_val = sum(relevant_vals)/len(relevant_vals)
            else:
                mean_val = 0
            labels.append(cur_llm)
            vals_list.append(mean_val)

        if not labels:
            continue

        plt.figure(figsize=(6,3.5))
        plt.bar(labels, vals_list, color='orange', alpha=0.7)
        plt.title(f"Aggregated {metric} by LLM\n(from {json_file_path})")
        plt.ylabel(metric)
        plt.grid(True, axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

    print("="*100)
    print(f"Total step 1: {len([x for x in step_metrics if x['step_index'] == 1])}")
    print(f"Total step 1.5: {len([x for x in step_metrics if x['step_index'] == 1.5])}")
    print(f"Total step 2: {len([x for x in step_metrics if x['step_index'] == 2])}")
    print(f"Total step 2.5: {len([x for x in step_metrics if x['step_index'] == 2.5])}")
    print(f"Total step 3: {len([x for x in step_metrics if x['step_index'] == 3])}")
    print(f"Total step 3.5: {len([x for x in step_metrics if x['step_index'] == 3.5])}")

    #print avg wellfare statistics
    print(f"Avg NWE of OUTCOME: {sum([x['nash_welfare'] for x in step_metrics if x['nash_welfare'] is not None and (x['action'] == 'ACCEPT' or x['action'] == 'WALK' or x['action'] == 'INVALID WALK')]) / 100}")
    print(f"Avg UWE of OUTCOME: {sum([x['utilitarian_welfare'] for x in step_metrics if x['utilitarian_welfare'] is not None and (x['action'] == 'ACCEPT' or x['action'] == 'WALK' or x['action'] == 'INVALID WALK')]) / 100}")
    print(f"Avg RWE of OUTCOME: {sum([x['rawlsian_welfare'] for x in step_metrics if x['rawlsian_welfare'] is not None and (x['action'] == 'ACCEPT' or x['action'] == 'WALK' or x['action'] == 'INVALID WALK')]) / 100}")
    print(f"Avg GINI of OUTCOME: {sum([x['gini_coefficient'] for x in step_metrics if x['gini_coefficient'] is not None and (x['action'] == 'ACCEPT' or x['action'] == 'WALK' or x['action'] == 'INVALID WALK')]) / 100}")
    print(f"Avg JAIN of OUTCOME: {sum([x['jain_fairness_index'] for x in step_metrics if x['jain_fairness_index'] is not None and (x['action'] == 'ACCEPT' or x['action'] == 'WALK' or x['action'] == 'INVALID WALK')]) / 100}")
    print(f"Avg MAD of OUTCOME: {sum([x['mean_absolute_difference'] for x in step_metrics if x['mean_absolute_difference'] is not None and (x['action'] == 'ACCEPT' or x['action'] == 'WALK' or x['action'] == 'INVALID WALK')]) / 100}")
    print(f"Avg SEC of OUTCOME: {sum([x['security_regret'] for x in step_metrics if x['security_regret'] is not None and (x['action'] == 'ACCEPT' or x['action'] == 'WALK' or x['action'] == 'INVALID WALK')]) / 100}")

    #counts of each action
    print(f"Total ACCEPT: {len([x for x in step_metrics if x['action'] == 'ACCEPT'])}")
    print(f"Total WALK: {len([x for x in step_metrics if x['action'] == 'WALK'])}")
    print(f"Total COUNTEROFFER: {len([x for x in step_metrics if x['action'] == 'COUNTEROFFER'])}")
    print(f"Total INVALID WALK: {len([x for x in step_metrics if x['action'] == 'INVALID WALK'])}")

    #average game length calculation via endings at each step index
    round_1_count = len([x for x in step_metrics if x['step_index'] == 1])
    round_1_5_count = len([x for x in step_metrics if x['step_index'] == 1.5])
    round_2_count = len([x for x in step_metrics if x['step_index'] == 2])
    round_2_5_count = len([x for x in step_metrics if x['step_index'] == 2.5])
    round_3_count = len([x for x in step_metrics if x['step_index'] == 3])
    round_3_5_count = len([x for x in step_metrics if x['step_index'] == 3.5])

    ended_at_1 = round_1_count - round_1_5_count
    ended_at_1_5 = round_1_5_count - round_2_count
    ended_at_2 = round_2_count - round_2_5_count
    ended_at_2_5 = round_2_5_count - round_3_count
    ended_at_3 = round_3_count - round_3_5_count
    ended_at_3_5 = round_3_5_count  
    total_games = ended_at_1 + ended_at_1_5 + ended_at_2 + ended_at_2_5 + ended_at_3 + ended_at_3_5
    weighted_sum = (
        ended_at_1 * 1.0
        + ended_at_1_5 * 1.5
        + ended_at_2 * 2.0
        + ended_at_2_5 * 2.5
        + ended_at_3 * 3.0
        + ended_at_3_5 * 3.5
    )
    avg_game_length = weighted_sum / total_games if total_games else 0

    print(f"Avg Game Length: {avg_game_length:.2f}")

    #avg value of the game for p1
    avg_value = sum([x['p1_undiscounted_value'] for x in step_metrics if x['p1_undiscounted_value'] is not None and (x['action'] == 'ACCEPT' or x['action'] == 'WALK' or x['action'] == 'INVALID WALK')]) / len([x for x in step_metrics if x['p1_undiscounted_value'] is not None and (x['action'] == 'ACCEPT' or x['action'] == 'WALK' or x['action'] == 'INVALID WALK')])
    print(f"Avg Value of Game for P1: {avg_value:.2f}")

    #avg value of the game for p2
    avg_value = sum([x['p2_undiscounted_value'] for x in step_metrics if x['p2_undiscounted_value'] is not None and (x['action'] == 'ACCEPT' or x['action'] == 'WALK' or x['action'] == 'INVALID WALK')]) / len([x for x in step_metrics if x['p2_undiscounted_value'] is not None and (x['action'] == 'ACCEPT' or x['action'] == 'WALK' or x['action'] == 'INVALID WALK')])
    print(f"Avg Value of Game for P2: {avg_value:.2f}")
   


import json
import os

def analyze_llm_all_circles(
    llm_name,
    base_path="/Users/gabesmithline/Desktop/caif_negotiation/experiments/4o_final",
    circles=range(7)
):
    """
    Reads JSON files for circles 0..6 for a given LLM and prints a table of:
      - avg_game_rounds (estimated by counting total steps / 2)
      - accept_count (total ACCEPT actions)
      - walk_count (total WALK actions)
      - invalid_walk_count (total INVALID WALK actions)

    We define each "round" as up to 2 steps (Player 1's move, then Player 2's move).
    Therefore, if a game ends earlier in a round, that final partial round still counts
    as 1 step, but since we are dividing total_steps by 2, on average it approximates
    the number of full (or partial) rounds that occurred.

    Assumes JSON files named like:
      <base_path>/<llm_name>_circle_{circle}.json
    Adjust if your actual filenames differ.
    """

    # We'll store per-circle results here, then print them at the end.
    circle_stats = {
        "avg_game_rounds": [],
        "accept_count": [],
        "walk_count": [],
        "invalid_walk_count": []
    }

    for circle in circles:
        json_file_path = os.path.join(base_path, f"{llm_name}_circle_{circle}.json")
        if not os.path.isfile(json_file_path):
            print(f"Warning: File not found: {json_file_path}")
            circle_stats["avg_game_rounds"].append(0.0)
            circle_stats["accept_count"].append(0)
            circle_stats["walk_count"].append(0)
            circle_stats["invalid_walk_count"].append(0)
            continue

        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_game_data = data.get("all_game_data", [])
        num_games = len(all_game_data)

        # If no games, store zeros
        if num_games == 0:
            circle_stats["avg_game_rounds"].append(0.0)
            circle_stats["accept_count"].append(0)
            circle_stats["walk_count"].append(0)
            circle_stats["invalid_walk_count"].append(0)
            continue

        total_steps_all_games = 0
        total_accept = 0
        total_walk = 0
        total_invalid_walk = 0

        # For each game in this circle
        for game_info in all_game_data:
            round_data = game_info.get("round_data", [])
            # number_of_steps is simply len(round_data)
            total_steps_all_games += len(round_data)

            # Count the actions
            for move in round_data:
                action = move.get("action", "").upper().strip()
                if action == "ACCEPT":
                    total_accept += 1
                elif action == "WALK":
                    total_walk += 1
                elif action == "INVALID WALK":
                    total_invalid_walk += 1
                # else: "COUNTEROFFER" or unknown -> ignore for these tallies

        avg_steps_per_game = total_steps_all_games / float(num_games)

        avg_game_rounds = avg_steps_per_game / 2.0

        # Store results for this circle
        circle_stats["avg_game_rounds"].append(avg_game_rounds)
        circle_stats["accept_count"].append(total_accept)
        circle_stats["walk_count"].append(total_walk)
        circle_stats["invalid_walk_count"].append(total_invalid_walk)


    print(f"\n=== Summary Stats for LLM: {llm_name} ===")
    circle_indices = list(circles)

    header = " " * 22 + "".join([f"{c:^10}" for c in circle_indices])
    print(header)

    for stat_name in ["avg_game_rounds", "accept_count", "walk_count", "invalid_walk_count"]:
        row_str = stat_name.ljust(22)
        for val in circle_stats[stat_name]:
            if stat_name == "avg_game_rounds":
                row_str += f"{val:^10.2f}"
            else:
                row_str += f"{val:^10}"
        print(row_str)



import math
from itertools import product

def compute_max_nash_welfare(item_counts, p1_valuations, p2_valuations):
    if len(item_counts) != len(p1_valuations) or len(item_counts) != len(p2_valuations):
        raise ValueError("item_counts, p1_valuations, p2_valuations must have the same length.")

    K = len(item_counts)
    max_nash = -1.0
    best_alloc = None
    outside_offer_player1 = np.random.randint(1, 101)
    outside_offer_player2 = np.random.randint(1, 101)
    ranges = [range(n_i + 1) for n_i in item_counts] 
    for allocation in product(*ranges):
        p1_util = 0.0
        p2_util = 0.0
        for i in range(K):
            x_i = allocation[i]
            n_i = item_counts[i]
            p1_util += x_i * p1_valuations[i]
            p2_util += (n_i - x_i) * p2_valuations[i]

        w = math.sqrt(max(p1_util, 0) * max(p2_util, 0))

        if w > max_nash:
            max_nash = w
            best_alloc = allocation

        #outside offer check
        if max_nash < np.sqrt(outside_offer_player1 * outside_offer_player2):
            max_nash = np.sqrt(outside_offer_player1 * outside_offer_player2)
            best_alloc = [0, 0, 0, 0, 0]

    return max_nash, list(best_alloc)



#print("Allocation to Player 1:", alloc)

def compare_llm_across_circles_with_stats_confint_and_distributions(
    base_path,
    circle_list,
    llm_name="unknown_llm",
    gamma=0.9
):
    """
    Reads multiple JSON files (one per circle) located at:
       base_path.format(circle=circle_number)
    e.g. "/Users/.../gemini_2.0_1_28_2025_circle_{circle}.json"
    
    For each circle in circle_list:
      1) Parses the data into 'step_metrics'.
      2) Aggregates so we can create comparison plots across circles for that single LLM
         (P1/P2 undiscounted values and fairness metrics).
      3) Computes final-outcome statistics (mean, stdev, variance, 95% confidence intervals)
         for P1 and P2 across each circle.
      4) Visualizes distributions of final payoffs with histograms (per circle)
         and grouped boxplots (across circles).

    We'll produce:
      • A single figure that plots avg P1 & P2 undiscounted values at each step for each circle.
      • For fairness metrics, a separate figure per metric with one line per circle.
      • Final-outcome statistics (mean, std dev, variance, 95% confidence intervals).
      • Histograms of final P1 & P2 payoffs per circle.
      • A grouped boxplot summarizing final outcomes across all circles.
    """

    import json
    import math
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import statistics

    # We'll store "step_metrics" for each circle in a dict: { circle_number: list_of_records }
    circle_to_metrics = {}
    # Final outcomes dictionary
    circle_final_p1_values = defaultdict(list)
    circle_final_p2_values = defaultdict(list)
    #nash welfare dict 
    circle_nash_welfare = defaultdict(list)
    circle_rawls_welfare = defaultdict(list)
    circle_utilitarian_welfare = defaultdict(list)
    circle_jain_fairness_index = defaultdict(list)
    circle_mean_absolute_difference = defaultdict(list)
    circle_security_regret = defaultdict(list)
    

    

    for c in circle_list:
        path_json = base_path.format(circle=c)
        with open(path_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_game_data = data.get("all_game_data", [])
        step_metrics = []

        
        for game_idx, game_info in enumerate(all_game_data):
            round_data = game_info.get("round_data", [])

            p1_valuations = None
            p2_valuations = None
            p1_outside = None
            p2_outside = None

            for move in round_data:
                prompt_text = move.get("prompt", "")
                if "You are Player 1" in prompt_text:
                    p = 1
                elif "You are Player 2" in prompt_text:
                    p = 2
                else:
                    p = None

                match_vals = re.search(
                    r"Your private values are\s+(\d+)\s+for item 1,\s*(\d+)\s+for item 2,\s*(\d+)\s+for item 3,\s*(\d+)\s+for item 4,\s*(\d+)\s+for item 5\.",
                    prompt_text
                )
                if match_vals:
                    v_list = [float(x) for x in match_vals.groups()]
                    if p == 1:
                        p1_valuations = v_list
                    elif p == 2:
                        p2_valuations = v_list

                match_out = re.search(r"Your outside offer value is\s+(\d+(\.\d+)?)\.", prompt_text)
                if match_out:
                    out_val = float(match_out.group(1))
                    if p == 1:
                        p1_outside = out_val
                    elif p == 2:
                        p2_outside = out_val
                
                item_counts = []
                match_item_counts = re.search(
                    r"There are\s+\d+\s+types of items.*?There are\s+(\d+)\s+units?\s+of\s+item\s+1,\s*(\d+)\s+units?\s+of\s+item\s+2,\s*(\d+)\s+units?\s+of\s+item\s+3,\s*(\d+)\s+units?\s+of\s+item\s+4,\s*(\d+)\s+units?\s+of\s+item\s+5",
                    prompt_text,
                    re.DOTALL
                )
                if match_item_counts:
                    item_counts = [int(x) for x in match_item_counts.groups()]

            last_co_val_p1 = None
            last_co_val_p2 = None
            current_round_num = 1
            i = 0
            while i < len(round_data):
                move = round_data[i]
                i += 1

                prompt_text = move.get("prompt", "")
                response_text = move.get("response", "")
                action = move.get("action", "")

                

                if "You are Player 1" in prompt_text:
                    cur_player = 1
                elif "You are Player 2" in prompt_text:
                    cur_player = 2
                else:
                    cur_player = None

                if cur_player == 1:
                    step_index = float(current_round_num)
                elif cur_player == 2:
                    step_index = float(current_round_num) + 0.5
                else:
                    step_index = float(current_round_num)

                try:
                    offer_match = re.search(
                        r"\"offer\"\s*:\s*\[\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*\]",
                        response_text
                    )
                except Exception as e:
                    print(f"Error parsing offer for game {game_idx} in circle {c}: {e}")
                    offer_match = None
                new_offer = None
                if offer_match:
                    new_offer = [float(x) for x in offer_match.groups()]

                def compute_offer_value(offer, valuations, player, current_player):
                    if player == current_player:
                        new_offer = []
                        for val in zip(item_counts, offer):
                            new_offer.append(val[0] - val[1])
                        offer = new_offer
                    return sum(o*v for o,v in zip(offer, valuations)) if (offer and valuations) else None
                if new_offer is not None:
                    p1_val_offer = compute_offer_value(new_offer, p1_valuations, 1, cur_player)
                    p2_val_offer = compute_offer_value(new_offer, p2_valuations, 2, cur_player)

                p1_val_final, p2_val_final = None, None
                if action == "COUNTEROFFER":
                    # update last_co_val
                    if cur_player == 1 and p1_val_offer is not None and p2_val_offer is not None:
                        last_co_val_p1 = p1_val_offer * gamma**(step_index - 1 if step_index % 1 != .5 else step_index - .5 - 1)
                        print(f"last_co_val_p1: {last_co_val_p1}")
                        print(f"step_index: {step_index}")
                        print(f"p2_val_offer: {p2_val_offer}")
                        last_co_val_p2 = p2_val_offer * gamma**(step_index - 1 if step_index % 1 != .5 else step_index - .5 - 1)
                    elif cur_player == 2 and p2_val_offer is not None and p1_val_offer is not None:
                        step_index = step_index - 0.5 + 1 #remove .5 and add 1 as move is realized in next step
                        print(f"step_index here: {step_index}")
                        last_co_val_p1 = p1_val_offer * gamma**(step_index if step_index % 1 != .5 else step_index - .5)
                        last_co_val_p2 = p2_val_offer * gamma**(step_index if step_index % 1 != .5 else step_index - .5)

                    p1_val_final = p1_val_offer
                    p2_val_final = p2_val_offer

                elif action == "ACCEPT":
                    if last_co_val_p1 is not None and last_co_val_p2 is not None:
                        p1_val_final = last_co_val_p1 
                        p2_val_final = last_co_val_p2
                    if p1_val_final is not None:
                        circle_final_p1_values[c].append((p1_val_final / np.dot(p1_valuations, item_counts)))
                    if p2_val_final is not None:
                        circle_final_p2_values[c].append((p2_val_final / np.dot(p2_valuations, item_counts)))

                    if p1_val_final is not None and p2_val_final and p2_val_final is not None:
                        max_nash, alloc = compute_max_nash_welfare(item_counts, p1_valuations, p2_valuations)
                        max_nash = max(max_nash, np.sqrt(p1_outside * p2_outside))

                        circle_nash_welfare[c].append(np.sqrt(p1_val_final * p2_val_final) / max_nash)
                        if np.sqrt(p1_val_final * p2_val_final) > max_nash:
                            print(f"accept circle {c} nash welfare: {np.sqrt(p1_val_final * p2_val_final)} > max nash welfare: {max_nash}")

                elif action in ["WALK","INVALID WALK"]:
                    # use outside offers
                    p1_val_final = p1_outside * gamma ** (step_index - 1 if step_index % 1 != .5 else step_index - .5 - 1)
                    if step_index == 1: 
                        p2_outside = np.random.randint(1, 101)
                        p2_val_final = p2_outside
                        p2_valuations = np.random.randint(1, 101, 5)
                    if step_index != 1:
                        p2_val_final = p2_outside * gamma ** (step_index - 1 if step_index % 1 != .5 else step_index - .5 - 1)
                    
                    if p1_val_final is not None:
                        circle_final_p1_values[c].append((p1_val_final / np.dot(item_counts, p1_valuations)))
                    if p2_val_final is not None:
                        if p2_valuations is not None:
                            circle_final_p2_values[c].append((p2_val_final / np.dot(item_counts, p2_valuations)))
                        else:
                            p2_valuations = np.random.randint(1, 101, 5)
                            circle_final_p2_values[c].append((p2_val_final / np.dot(item_counts, p2_valuations)))

                    if p1_val_final is not None and p2_val_final is not None:
                        max_nash, alloc = compute_max_nash_welfare(item_counts, p1_valuations, p2_valuations)
                        max_nash = max(max_nash, np.sqrt(p1_outside * p2_outside))

                    
                        circle_nash_welfare[c].append(np.sqrt(p1_val_final * p2_val_final) / max_nash)
                        if np.sqrt(p1_val_final * p2_val_final) > max_nash:
                            print(f"walk circle {c} nash welfare: {np.sqrt(p1_val_final * p2_val_final)} > max nash welfare: {max_nash}")
               
                def compute_gini(a,b):
                    total = a+b
                    if total <= 0: return 0.0
                    return abs(a-b)/(4*total)

                def compute_nash(a,b):
                    x1 = max(a,0)
                    x2 = max(b,0)
                    if cur_player == 1:
                        return math.sqrt(x1*x2) * gamma ** (step_index)
                    else:
                        return math.sqrt(x1*x2) * gamma ** (step_index-0.5)

                def compute_rawls(a,b):
                    return min(a,b)

                def compute_util(a,b):
                    return a+b

                def compute_jain(a,b):
                    x1 = max(a,0)
                    x2 = max(b,0)
                    s = x1 + x2
                    q = x1*x1 + x2*x2
                    if q == 0: return 0
                    return (s*s)/(2.0*q)

                def compute_mad(a,b):
                    return abs(a-b)/2.0

                if (p1_val_final is not None) and (p2_val_final is not None):
                    step_metrics.append({
                        "circle_name": f"circle_{c}",
                        "llm_name": llm_name,
                        "game_index": game_idx,
                        "player": cur_player,
                        "step_index": step_index,
                        "action": action,
                        "p1_undiscounted_value": p1_val_final,
                        "p2_undiscounted_value": p2_val_final,
                        "gini_coefficient": compute_gini(p1_val_final, p2_val_final),
                        "nash_welfare": compute_nash(p1_val_final, p2_val_final),
                        "rawlsian_welfare": compute_rawls(p1_val_final, p2_val_final),
                        "utilitarian_welfare": compute_util(p1_val_final, p2_val_final),
                        "jain_fairness_index": compute_jain(p1_val_final, p2_val_final),
                        "mean_absolute_difference": compute_mad(p1_val_final, p2_val_final),
                        "security_regret": None  
                    })
                else:
                    step_metrics.append({
                        "circle_name": f"circle_{c}",
                        "llm_name": llm_name,
                        "game_index": game_idx,
                        "player": cur_player,
                        "step_index": step_index,
                        "action": action,
                        "p1_undiscounted_value": None,
                        "p2_undiscounted_value": None,
                        "gini_coefficient": None,
                        "nash_welfare": None,
                        "rawlsian_welfare": None,
                        "utilitarian_welfare": None,
                        "jain_fairness_index": None,
                        "mean_absolute_difference": None,
                        "security_regret": None
                    })

                if cur_player == 2:
                    current_round_num += 1

        circle_to_metrics[c] = step_metrics

   
    from matplotlib.cm import get_cmap
    color_map = get_cmap("tab10")

    print(f"Nash Welfare Statistics:")
    for c in circle_nash_welfare:
        holder = circle_nash_welfare[c]
        
        holder = np.array(holder)
        print(np.mean(holder))
        
       
        variance = np.var(holder, ddof=1)
        std_dev = np.std(holder, ddof=1)
        p25 = np.percentile(holder, 25)  # 25th percentile
        p75 = np.percentile(holder, 75)  # 75th percentile
        mean_val = np.mean(holder)

        print(f"circle {c} variance: {variance}")
        print(f"circle {c} std: {std_dev}")
        print(f"circle {c} 25th percentile: {p25}")
        print(f"circle {c} 75th percentile: {p75}")
        print(f"circle {c} mean / global max nash welfare: {mean_val}")
        

        #plot
        #plot
        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(holder, 100, density=True, facecolor='blue', alpha=0.75, label='Grade Distribution') #NOTE: change the number of bins to see the effect on curve
        y = ((1 / (np.sqrt(2 * np.pi) * std_dev)) *
            np.exp(-0.5 * (1 / std_dev * (bins - mean_val))**2)) #
        plt.plot(bins, y, 'r--', label='Normal Distribution Fit')
        plt.title('Normal Distribution')
        plt.xlabel('X')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

    all_steps_union = set()
    for c in circle_to_metrics:
        for rec in circle_to_metrics[c]:
            all_steps_union.add(rec["step_index"])
    all_steps_sorted = sorted(list(all_steps_union))

    plt.figure(figsize=(9,5))
    for idx, c in enumerate(sorted(circle_list)):
        step_lookup = defaultdict(lambda: {"p1": [], "p2": []})
        step_metrics = circle_to_metrics[c]
        for rec in step_metrics:
            sidx = rec["step_index"]
            valp1 = rec["p1_undiscounted_value"]
            valp2 = rec["p2_undiscounted_value"]
            nashwelfare = rec["nash_welfare"]
          
            if valp1 is not None:
                step_lookup[sidx]["p1"].append(valp1)
            if valp2 is not None:
                step_lookup[sidx]["p2"].append(valp2)

        sorted_steps = sorted(step_lookup.keys())
        p1_vals, p2_vals = [], []
        for s in sorted_steps:
            arr_p1 = step_lookup[s]["p1"]
            arr_p2 = step_lookup[s]["p2"]
            p1_avg = sum(arr_p1)/len(arr_p1) if arr_p1 else None
            p2_avg = sum(arr_p2)/len(arr_p2) if arr_p2 else None
            p1_vals.append(p1_avg)
            p2_vals.append(p2_avg)

        circle_color = color_map(idx % 10)
        lab_p1 = f"Circle {c} P1"
        lab_p2 = f"Circle {c} P2"

        plt.plot(
            sorted_steps,
            p1_vals,
            color=circle_color,
            linestyle='-',
            marker='o',
            label=lab_p1
        )
        plt.plot(
            sorted_steps,
            p2_vals,
            color=circle_color,
            linestyle='--',
            marker='s',
            label=lab_p2
        )

    plt.title(f"Comparison of P1/P2 Undiscounted Values Across Circles (LLM: {llm_name})")
    plt.xlabel("Step index")
    plt.ylabel("Undiscounted Value")
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

   
    metrics_to_plot = [
        "gini_coefficient",
        "nash_welfare",
        "rawlsian_welfare",
        "utilitarian_welfare",
        "jain_fairness_index",
        "mean_absolute_difference"
    ]

    for metric in metrics_to_plot:
        plt.figure(figsize=(7,4))
        for idx, c in enumerate(sorted(circle_list)):
            step_metrics = circle_to_metrics[c]
            circle_data = [r for r in step_metrics if r["circle_name"] == f"circle_{c}"]
            if not circle_data:
                continue

            # Build step -> list of metric-values
            step_lookup_metrics = defaultdict(list)
            for rec in circle_data:
                sidx = rec["step_index"]
                val = rec.get(metric, None)
                if val is not None:
                    step_lookup_metrics[sidx].append(val)

            if not step_lookup_metrics:
                continue

            sorted_steps = sorted(step_lookup_metrics.keys())
            yvals = []
            for s in sorted_steps:
                arr = step_lookup_metrics[s]
                yvals.append(sum(arr)/len(arr))

            circle_color = color_map(idx % 10)
            plt.plot(
                sorted_steps,
                yvals,
                marker='o',
                linestyle='-',
                color=circle_color,
                label=f"Circle {c}"
            )

        plt.title(f"Comparison of {metric} by Step\n(LLM: {llm_name})")
        plt.xlabel("Step index")
        plt.ylabel(metric)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    
    print(f"\n{'='*80}")
    print("Final-Outcome Statistics (ACCEPT/WALK/INVALID WALK) Per Circle w/ 95% CIs")
    print(f"{'='*80}\n")

    def compute_confidence_interval(vals, confidence=0.95):
        """
        Returns (mean, lower_bound, upper_bound) for the approximate
        confidence interval of the mean, using normal approximation:
            mean +/- z*(stdev/sqrt(n))  with z ~ 1.96 for 95% if n >= ~30
        If n < 2, returns mean, mean, mean (no real interval).
        """
        n = len(vals)
        if n < 2:
            if n == 0:
                return (None, None, None)
            return (vals[0], vals[0], vals[0])

        mean_val = statistics.mean(vals)
        stdev_val = statistics.pstdev(vals)
        z = 1.96  # 95% CI for large n 
        se = stdev_val / math.sqrt(n)
        margin = z * se
        lower = mean_val - margin
        upper = mean_val + margin
        return (mean_val, lower, upper)

    for c in sorted(circle_list):
        p1_vals = circle_final_p1_values[c]
        p2_vals = circle_final_p2_values[c]
        n_games = len(p1_vals)

        

        print(f"Circle {c}: {n_games} final outcomes recorded")
        if n_games == 0:
            print(" No final records found.\n")
            continue

        # P1 stats
        p1_mean = statistics.mean(p1_vals)
        p1_stdev = statistics.pstdev(p1_vals) if n_games>1 else 0.0
        p1_var   = statistics.pvariance(p1_vals) if n_games>1 else 0.0
        p1_m, p1_lo, p1_hi = compute_confidence_interval(p1_vals, confidence=0.95)

        # P2 stats
        p2_vals_filtered = [val for val in p2_vals if val is not None]
        if len(p2_vals_filtered) > 0:
            p2_mean = statistics.mean(p2_vals_filtered)
            p2_stdev = statistics.pstdev(p2_vals_filtered) if len(p2_vals_filtered) > 1 else 0.0
            p2_var = statistics.pvariance(p2_vals_filtered) if len(p2_vals_filtered) > 1 else 0.0
            p2_m, p2_lo, p2_hi = compute_confidence_interval(p2_vals_filtered, confidence=0.95)
        else:
            p2_mean, p2_stdev, p2_var = 0.0, 0.0, 0.0
            p2_m, p2_lo, p2_hi = None, None, None

        print(f"  P1 final payoff:")
        print(f"     mean = {p1_mean:.2f}, stdev = {p1_stdev:.2f}, variance = {p1_var:.2f}")
        if p1_m is not None and p1_lo is not None and p1_hi is not None:
            print(f"     95% CI = [{p1_lo:.2f}, {p1_hi:.2f}]")

        print(f"  P2 final payoff:")
        print(f"     mean = {p2_mean:.2f}, stdev = {p2_stdev:.2f}, variance = {p2_var:.2f}")
        if p2_m is not None and p2_lo is not None and p2_hi is not None:
            print(f"     95% CI = [{p2_lo:.2f}, {p2_hi:.2f}]")
        print()

   
    fig, axes = plt.subplots(nrows=len(circle_list), ncols=2, figsize=(10, 3*len(circle_list)))
    if len(circle_list) == 1:
        axes = [axes] 

    for idx, c in enumerate(sorted(circle_list)):
        # Filter out None values to prevent TypeError in hist
        p1_vals_filtered = [val for val in circle_final_p1_values[c] if val is not None]
        p2_vals_filtered = [val for val in circle_final_p2_values[c] if val is not None]

        ax_left = axes[idx][0]
        ax_right = axes[idx][1]

        ax_left.hist(p1_vals_filtered, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax_left.set_title(f"Circle {c} - P1 Dist")
        ax_left.set_xlabel("P1 payoff")
        ax_left.set_ylabel("Freq")

        ax_right.hist(p2_vals_filtered, bins=15, color='darkorange', edgecolor='black', alpha=0.7)
        ax_right.set_title(f"Circle {c} - P2 Dist")
        ax_right.set_xlabel("P2 payoff")
        ax_right.set_ylabel("Freq")

    plt.tight_layout()
    plt.show()

   
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

    p1_box_data = []
    for c in sorted(circle_list):
        filtered_data = [val for val in circle_final_p1_values[c] if val is not None]
        p1_box_data.append(filtered_data)
    ax1.boxplot(p1_box_data, patch_artist=True)
    ax1.set_xticks(range(1, len(circle_list)+1))
    ax1.set_xticklabels([str(c) for c in sorted(circle_list)])
    ax1.set_title("P1 Final Payoff Distribution Across Circles")
    ax1.set_xlabel("Circle")
    ax1.set_ylabel("P1 Payoff")

    p2_box_data = []
    for c in sorted(circle_list):
        filtered_data = [val for val in circle_final_p2_values[c] if val is not None]
        p2_box_data.append(filtered_data)
    ax2.boxplot(p2_box_data, patch_artist=True)
    ax2.set_xticks(range(1, len(circle_list)+1))
    ax2.set_xticklabels([str(c) for c in sorted(circle_list)])
    ax2.set_title("P2 Final Payoff Distribution Across Circles")
    ax2.set_xlabel("Circle")
    ax2.set_ylabel("P2 Payoff")

    plt.tight_layout()
    plt.show()




import json
import math
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def compare_llms_for_circle(
    base_path1,
    base_path2,
    circle_number,
    llm_list,
    gamma=0.9
):


    def parse_player_id(prompt_text):
        if "You are Player 1" in prompt_text:
            return 1
        elif "You are Player 2" in prompt_text:
            return 2
        else:
            return None

    def parse_valuations_from_prompt(prompt_text):
        pattern_vals = re.compile(
            r"Your private values are\s+(\d+)\s+for item 1,\s*(\d+)\s+for item 2,\s*(\d+)\s+for item 3,\s*(\d+)\s+for item 4,\s*(\d+)\s+for item 5\."
        )
        match_vals = pattern_vals.search(prompt_text)
        if match_vals:
            return [float(x) for x in match_vals.groups()]
        return None

    def parse_outside_offer_from_prompt(prompt_text):
        pattern_out = re.compile(r"Your outside offer value is\s+(\d+(\.\d+)?)\.")
        match_out = pattern_out.search(prompt_text)
        if match_out:
            return float(match_out.group(1))
        return None

    def parse_items_from_prompt(prompt_text):
        pattern_items = re.compile(
            r"There are\s+(\d+)\s+units of item 1,\s+(\d+)\s+units of item 2,\s+(\d+)\s+units of item 3,\s+(\d+)\s+units of item 4,\s+(\d+)\s+units of item 5 to divide\."
        )
        match_items = pattern_items.search(prompt_text)
        if match_items:
            return [int(x) for x in match_items.groups()]
        return None

    def parse_offer_from_response(response_text):
        pattern = re.compile(
            r"\"offer\"\s*:\s*\[\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*\]"
        )
        match = pattern.search(response_text)
        if match:
            return [float(g) for g in match.groups()]
        return None

    def compute_dotproduct(offer, valuations):
        return sum(o * v for o, v in zip(offer, valuations))

    def compute_gini(p1_val, p2_val):
        total = p1_val + p2_val
        if total <= 0:
            return 0.0
        return abs(p1_val - p2_val) / (4 * total)

    def compute_nash(p1_val, p2_val):
        x1 = max(0.0, p1_val)
        x2 = max(0.0, p2_val)
        return math.sqrt(x1 * x2)

    def compute_rawls(p1_val, p2_val):
        return min(p1_val, p2_val)

    def compute_utilitarian(p1_val, p2_val):
        return p1_val + p2_val

    def compute_jain_fairness(p1_val, p2_val):
        x1 = max(0.0, p1_val)
        x2 = max(0.0, p2_val)
        s = x1 + x2
        q = x1*x1 + x2*x2
        if q == 0:
            return 0.0
        return (s*s)/(2.0*q)

    def compute_mean_absolute_diff(p1_val, p2_val):
        return abs(p1_val - p2_val) / 2.0

    def compute_security_regret(my_val, outside_val):
        return max(0.0, outside_val - my_val)

    def check_5_mistakes(current_action,
                         my_val_cur,
                         my_val_prev,
                         out_offer,
                         new_offer,
                         total_items):
        mistakes = {k: False for k in range(1, 6)}

        def approx_lt(a, b):
            return (a is not None) and (b is not None) and (round(a, 6) < round(b, 6))

        if current_action == "COUNTEROFFER":
            if new_offer is not None and total_items is not None:
                if all(q == 0 for q in new_offer) or all(q == t for q, t in zip(new_offer, total_items)):
                    mistakes[3] = True
            if my_val_cur is not None and my_val_prev is not None:
                if approx_lt(my_val_cur, my_val_prev):
                    mistakes[1] = True
            if my_val_cur is not None and out_offer is not None:
                if approx_lt(my_val_cur, out_offer):
                    mistakes[2] = True
        elif current_action == "ACCEPT":
            if my_val_cur is not None and out_offer is not None:
                if approx_lt(my_val_cur, out_offer):
                    mistakes[4] = True
        elif current_action == "WALK":
            if my_val_cur is not None and out_offer is not None:
                if my_val_cur > out_offer:
                    mistakes[5] = True

        return mistakes


    llm_to_metrics = {}

    for llm in llm_list:
        if llm == "4o":
            file_path = base_path2.format(llm=llm, circle=circle_number)
        else:
            file_path = base_path1.format(llm=llm, circle=circle_number)
        print(f"Reading file {file_path} for LLM: {llm}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"WARNING: file not found -> {file_path}")
            llm_to_metrics[llm] = []
            continue

        all_game_data = data.get("all_game_data", [])
        step_metrics = []
        
        for game_idx, game_info in enumerate(all_game_data):
            round_data = game_info.get("round_data", [])
            p1_valuations, p2_valuations = None, None
            p1_outside, p2_outside = None, None
            total_items = None

            for move in round_data:
                prompt_text = move.get("prompt", "")
                p = parse_player_id(prompt_text)  
                v = parse_valuations_from_prompt(prompt_text)
                out_val = parse_outside_offer_from_prompt(prompt_text)
                items_list = parse_items_from_prompt(prompt_text)
                if items_list is not None:
                    total_items = items_list
                if p == 1:
                    if v is not None:
                        p1_valuations = v
                    if out_val is not None:
                        p1_outside = out_val
                elif p == 2:
                    if v is not None:
                        p2_valuations = v
                    if out_val is not None:
                        p2_outside = out_val

            last_counter_offer_val_p1 = None
            last_counter_offer_val_p2 = None

            prev_val_p1, prev_val_p2 = None, None
            current_round_num = 1

            i = 0
            while i < len(round_data):
                move = round_data[i]
                i += 1
                prompt_text = move.get("prompt", "")
                response_text = move.get("response", "")
                action = move.get("action", "")
                try:
                    cur_player = parse_player_id(prompt_text)
                except:
                    cur_player = None

                new_offer = parse_offer_from_response(response_text)
                if cur_player == 1:
                    step_index = float(current_round_num)
                elif cur_player == 2:
                    step_index = float(current_round_num) + 0.5
                else:
                    step_index = float(current_round_num)  # fallback

                p1_offer_val, p2_offer_val = None, None
                if (p1_valuations and p2_valuations 
                    and new_offer and len(new_offer) == len(p1_valuations)):
                    p1_offer_val = compute_dotproduct(new_offer, p1_valuations)
                    p2_offer_val = compute_dotproduct(new_offer, p2_valuations)

                if action == "COUNTEROFFER":
                    if cur_player == 1:
                        last_counter_offer_val_p1 = p1_offer_val
                        last_counter_offer_val_p2 = p2_offer_val
                    elif cur_player == 2:
                        last_counter_offer_val_p1 = p1_offer_val
                        last_counter_offer_val_p2 = p2_offer_val

                if cur_player == 1:
                    my_val_cur = p1_offer_val
                    my_val_prev = prev_val_p1
                    out_offer = p1_outside
                elif cur_player == 2:
                    my_val_cur = p2_offer_val
                    my_val_prev = prev_val_p2
                    out_offer = p2_outside
                else:
                    my_val_cur = None
                    my_val_prev = None
                    out_offer = None

                mistakes = check_5_mistakes(
                    action, my_val_cur, my_val_prev, out_offer, new_offer, total_items
                )

                # update prev_val
                if cur_player == 1 and my_val_cur is not None:
                    prev_val_p1 = my_val_cur
                elif cur_player == 2 and my_val_cur is not None:
                    prev_val_p2 = my_val_cur

                p1_val_final, p2_val_final = None, None
                if action == "COUNTEROFFER":
                    if p1_offer_val is not None and p2_offer_val is not None:
                        p1_val_final = p1_offer_val * gamma**step_index
                        p2_val_final = p2_offer_val * gamma**step_index
                elif action == "ACCEPT":
                    if last_counter_offer_val_p1 is not None and last_counter_offer_val_p2 is not None:
                        p1_val_final = last_counter_offer_val_p1 * gamma**step_index
                        p2_val_final = last_counter_offer_val_p2 * gamma**step_index
                elif action in ["WALK", "INVALID WALK"]:
                    if p1_outside is not None and p2_outside is not None:
                        p1_val_final = p1_outside * gamma**step_index
                        p2_val_final = p2_outside * gamma**step_index

                if p1_val_final is not None and p2_val_final is not None:
                    gini_val = compute_gini(p1_val_final, p2_val_final)
                    nash_val = compute_nash(p1_val_final, p2_val_final)
                    rawls_val = compute_rawls(p1_val_final, p2_val_final)
                    util_val = compute_utilitarian(p1_val_final, p2_val_final)
                    jain_val = compute_jain_fairness(p1_val_final, p2_val_final)
                    mad_val = compute_mean_absolute_diff(p1_val_final, p2_val_final)
                    
                    if cur_player == 1 and p1_outside is not None:
                        sec_reg = compute_security_regret(p1_val_final, p1_outside)
                    elif cur_player == 2 and p2_outside is not None:
                        sec_reg = compute_security_regret(p2_val_final, p2_outside)
                    else:
                        sec_reg = None
                else:
                    gini_val, nash_val, rawls_val = None, None, None
                    util_val, jain_val, mad_val, sec_reg = None, None, None, None

                concession_size = None
                if (action == "COUNTEROFFER") and (my_val_prev is not None) and (my_val_cur is not None):
                    diff_val = my_val_prev - my_val_cur
                    if diff_val > 0:
                        concession_size = diff_val

                step_metrics.append({
                    "circle_name": f"circle_{circle_number}",
                    "llm_name": llm,
                    "game_index": game_idx,
                    "round_number": current_round_num,
                    "player": cur_player,
                    "step_index": step_index,
                    "action": action,

                    "p1_undiscounted_value": p1_val_final,
                    "p2_undiscounted_value": p2_val_final,

                    "gini_coefficient": gini_val,
                    "nash_welfare": nash_val,
                    "rawlsian_welfare": rawls_val,
                    "utilitarian_welfare": util_val,
                    "jain_fairness_index": jain_val,
                    "mean_absolute_difference": mad_val,
                    "security_regret": sec_reg,
                    "concession_size": concession_size,

                    "mistake_1": mistakes[1],
                    "mistake_2": mistakes[2],
                    "mistake_3": mistakes[3],
                    "mistake_4": mistakes[4],
                    "mistake_5": mistakes[5],
                })

                if cur_player == 2:
                    current_round_num += 1

        llm_to_metrics[llm] = step_metrics
 
    plt.figure(figsize=(9,5))
    color_map = plt.cm.get_cmap('tab10')  

    all_steps_union = set()
    for llm in llm_list:
        all_steps_union |= set([rec["step_index"] for rec in llm_to_metrics[llm]])


    all_steps_sorted = sorted(list(all_steps_union))

    for idx, llm in enumerate(llm_list):
        step_lookup = defaultdict(lambda: {"p1": [], "p2": []})
        these_metrics = llm_to_metrics[llm]
        for r in these_metrics:
            if r["p1_undiscounted_value"] is not None:
                step_lookup[r["step_index"]]["p1"].append(r["p1_undiscounted_value"])
            if r["p2_undiscounted_value"] is not None:
                step_lookup[r["step_index"]]["p2"].append(r["p2_undiscounted_value"])
        
        steps_for_llm = sorted(step_lookup.keys())
        p1_y, p2_y = [], []
        for s in steps_for_llm:
            p1_list = step_lookup[s]["p1"]
            p2_list = step_lookup[s]["p2"]
            p1_avg = sum(p1_list)/len(p1_list) if p1_list else None
            p2_avg = sum(p2_list)/len(p2_list) if p2_list else None
            p1_y.append((s, p1_avg))
            p2_y.append((s, p2_avg))

        color = color_map(idx % 10)
        p1_steps = [x[0] for x in p1_y]
        p1_vals  = [x[1] for x in p1_y]

        p2_steps = [x[0] for x in p2_y]
        p2_vals  = [x[1] for x in p2_y]

        plt.plot(p1_steps, p1_vals, marker='o', color=color, linestyle='-', label=f"{llm} P1")
        plt.plot(p2_steps, p2_vals, marker='s', color=color, linestyle='--', label=f"{llm} P2")

    plt.title(f"Comparison of P1/P2 Values Across LLMs (Circle: {circle_number})")
    plt.xlabel("Step Index")
    plt.ylabel("Undiscounted Value")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


    metrics_to_plot = [
        "gini_coefficient",
        "nash_welfare",
        "rawlsian_welfare",
        "utilitarian_welfare",
        "jain_fairness_index",
        "mean_absolute_difference",
        "security_regret",
        "concession_size"
    ]

    for metric in metrics_to_plot:
        plt.figure(figsize=(7,4))
        for idx, llm in enumerate(llm_list):
            these_metrics = llm_to_metrics[llm]
            step_lookup = defaultdict(list)
            for rec in these_metrics:
                sidx = rec["step_index"]
                val = rec.get(metric, None)
                if val is not None:
                    step_lookup[sidx].append(val)
            if not step_lookup:
                continue

            sorted_steps = sorted(step_lookup.keys())
            yvals = [(s, sum(step_lookup[s])/len(step_lookup[s])) for s in sorted_steps]

            color = color_map(idx % 10)
            plt.plot(
                [pt[0] for pt in yvals],
                [pt[1] for pt in yvals],
                marker='o',
                linestyle='-',
                color=color,
                label=llm
            )

        plt.title(f"Comparison of {metric} by Step\n(Circle: {circle_number})")
        plt.xlabel("Step Index")
        plt.ylabel(metric)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

