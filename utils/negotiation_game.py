from game_runner import NegotitaionGame
import agents.llm_agent as llm_agent
import numpy as np
import pandas as pd
import sys
sys.path.append('../caif_negotiation/')
from test_game_eval import *
from eval.metrics import *
import time
import pandas as pd
import numpy as np
from utils.helpers import *
import time
import numpy as np
from eval.game_data import GameData 
import pickle
import json

def run_game(circle1: int, circle2: int, games: int, max_rounds: int, date: str, game_title: str, llm_model_p1: str, llm_model_p2: str, discount: float):
    """
    Runs a series of negotiation games for a specific circle, tracking comprehensive metrics.

    Args:
        circle (int): The circle parameter influencing allocation strategies.
        games (int): Number of games to simulate.
        max_rounds (int): Maximum number of rounds per game.
        date (str): Date identifier for result files.
        game_title (str): Title identifier for the game series.
        llm_model_p1 (str): Type of LLM agent being used (e.g., "openai_4o").
        llm_model_p2 (str): Type of LLM agent being used (e.g., "openai_o3_mini").
    """
    all_game_data = []
    completed_games = 0
    attempts = 0
    max_attempts = games + 50  # Set a reasonable upper limit to prevent infinite loops
    
    while completed_games < games and attempts < max_attempts:
        attempts += 1
        if (attempts) % 10 == 0:
            print(f"Game attempt {attempts}, completed {completed_games} of {games}")
            sleep_duration = 2 * np.random.randint(55, 60)  # Sleep for ~2 minutes
            print(f"Sleeping for {sleep_duration} seconds to respect rate limits.")
            time.sleep(sleep_duration)
            
        game = NegotitaionGame(
            player1_agent=llm_agent.LLMAgent(llm_type=llm_model_p1, model=llm_model_p1, player_num=0),
            player2_agent=llm_agent.LLMAgent(llm_type=llm_model_p2, model=llm_model_p2, player_num=1),
            num_items=5,
            item_value_range=[1, 101],
            gamma=discount,
            max_rounds=max_rounds,
            circle1=circle1,
            circle2=circle2 
        )
       
        pareto_front = compute_pareto_frontier(
            game.player_values[0],
            game.player_values[1],
            game.num_items,
            game.items,
            game.outside_offer_values
        )

        allocations_less_than_outside_offer = None
        if circle1 in (5, 6) or circle2 in (5, 6):
            allocations_less_than_outside_offer = []

            allocation_p1 = find_allocation_less_than_outside_offer_dp(
                items=game.items,
                player_valuations=game.player_values[0],
                outside_offer=game.outside_offer_values[0],
                player_num=1
            )
            if allocation_p1:
                allocations_less_than_outside_offer.append({
                    'player': 1,
                    'allocation': list(allocation_p1.values())
                })
            else:
                allocations_less_than_outside_offer.append({
                    'player': 1,
                    'allocation': [0] * game.num_items
                })
                print(f"[INFO] No feasible < outside_offer allocation for Player 1 in Game attempt {attempts}.")

            # Find allocations where Player 2's utility is less than their outside offer
            allocation_p2 = find_allocation_less_than_outside_offer_dp(
                items=game.items,
                player_valuations=game.player_values[1],
                outside_offer=game.outside_offer_values[1],
                player_num=2
            )
            if allocation_p2:
                allocations_less_than_outside_offer.append({
                    'player': 2,
                    'allocation': list(allocation_p2.values())
                })
            else:
                allocations_less_than_outside_offer.append({
                    'player': 2,
                    'allocation': [0] * game.num_items
                })
                print(f"[INFO] No feasible < outside_offer allocation for Player 2 in Game attempt {attempts}.")

            print(f"[DEBUG] Game attempt {attempts} allocations_less_than_outside_offer: {allocations_less_than_outside_offer}")

        print(f"[DEBUG] game.items: {game.items}")
        print(f"[DEBUG] allocations_less_than_outside_offer: {allocations_less_than_outside_offer}")

        game_history = GameHistory(
            agent_1_name="Agent1",
            agent_2_name="Agent2",
            num_items=game.num_items,
            items=torch.tensor(game.items),
            agent_1_values=torch.tensor(game.player_values[0]),
            agent_2_values=torch.tensor(game.player_values[1]),
            agent_1_outside_value=game.outside_offer_values[0],
            agent_2_outside_value=game.outside_offer_values[1]
        )
        game_history.agent_1_offers = []
        game_history.agent_2_offers = []

     
        game_data = GameData(
            circle=(circle1, circle2),
            date=date,
            agent1=f"Agent1_{llm_model_p1}",
            agent2=f"Agent2_{llm_model_p2}"
        )

        print(f"[INFO] Starting Game attempt {attempts} (completed {completed_games} of {games}) for Circle {circle1 if game.current_player == 0 else circle2}.")

        # Flag to track if this game had an API failure
        had_api_failure = False

        while game.in_progress and not had_api_failure:
            # Sleep to simulate thinking time and rate-limit API calls
            sleep_duration = circle1 if game.current_player == 0 else circle2 + .5  # Adjust based on desired rate-limiting
            print(f"[DEBUG] Sleeping for {sleep_duration} seconds before next step.")
            sleep_duration = np.random.randint(sleep_duration, sleep_duration + 10)
            time.sleep(sleep_duration)

            # Determine current step, round, and player
            current_step = len(game.history[0]) + len(game.history[1]) + 1
            current_round = (current_step - 1) // 2 + 1
            current_player = 1 if current_step % 2 == 1 else 2
            game.current_round = current_round

            print("\n" + "=" * 80)
            print(f"Game attempt {attempts}, Round {current_round}, Player {current_player}'s turn (Step {current_step})")
            print("=" * 80)

            current_allocation_example = None
            if circle1 in (5, 6) or circle2 in (5, 6) and allocations_less_than_outside_offer is not None:
                if current_player == 1:
                    current_allocation_example = allocations_less_than_outside_offer[0]['allocation']
                elif current_player == 2:
                    current_allocation_example = allocations_less_than_outside_offer[1]['allocation']

            print(f"[DEBUG] Current allocation example type: {type(current_allocation_example)}")

            game.step(example_offer_less_than_outside_offer_self=current_allocation_example)
            
            # Check if an API failure occurred during this step
            current_agent = game.players[current_player - 1]
            if hasattr(current_agent, 'api_failure') and current_agent.api_failure:
                print(f"[WARNING] API failure detected in step {current_step}. Skipping this game.")
                had_api_failure = True
                break
                
            action_played = game.players[current_player - 1].action.upper()

            game_data.add_round_data(
                    prompt=game.players[current_player - 1].current_prompt,
                    response=game.players[current_player - 1].current_response,  # Assuming response includes the agent's textual response
                    action=action_played
                )

            if "WALK" in action_played or "ACCEPT" in action_played:
                game.in_progress = False

        # Only add this game to results if there was no API failure
        if not had_api_failure:
            all_game_data.append(game_data)
            completed_games += 1
            print(f"[INFO] Game {completed_games} completed successfully.")
        else:
            print(f"[INFO] Game attempt {attempts} skipped due to API failure.")

    if completed_games < games:
        print(f"[WARNING] Only completed {completed_games} of {games} requested games after {attempts} attempts due to API failures.")
    
    print("HERE IS THE DATA")
    all_data = {
        "date": date,
        "circle_p1": circle1,
        "circle_p2": circle2,
        "all_game_data": [gd.to_dict() for gd in all_game_data]
    }
    all_games_filename = f'all_game_data_{date}_{completed_games}_{game_title}_circle_p1_{circle1}_circle_p2_{circle2}.json'
    with open(all_games_filename, "w") as f:
        json.dump(all_data, f, indent=4)
        #json.pickle(all_data, f)
    print(f"[INFO] Saved all GameData to JSON file: {all_games_filename}.")

    #save to pickle optinally
    all_games_filename_pkl = f'all_game_data_{date}_{completed_games}_{game_title}_circle_p1_{circle1}_circle_p2_{circle2}.pkl'
    with open(all_games_filename_pkl, "wb") as pf:
        pickle.dump(all_data, pf)
    print(f"[INFO] Saved all GameData as a pickle to {all_games_filename_pkl}.")