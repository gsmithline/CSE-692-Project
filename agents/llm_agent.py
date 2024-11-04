from agents.agent import Agent
from offer import Offer
import json

from agents.agent import Agent
from offer import Offer
import json

class LLMAgent(Agent):
    def __init__(self, llm_type="llama", api_key=None, player_num=None):
        super().__init__(llm_type=llm_type, api_key=api_key)
        self.player_num = player_num
        self.llm_type = llm_type
        
    def give_offer(self, prompt: str) -> Offer | bool:
        system_prompt = """You are an AI negotiator participating in a negotiation game. 

        You must respond in one of these formats ONLY:
        1. {"action": "ACCEPT"} - to accept the current offer
        2. {"action": "WALK"} - to walk away from negotiations
        3. {"action": "COUNTEROFFER", "offer": [n1, n2, ...]} - where n1, n2, ... are numbers representing your counteroffer. You recieve the items left over from your offer to the other player.

        Ensure your response is valid JSON and matches one of these exact formats."""
        
        if self.llm_type == "llama":
            api_request = {
                "model": "llama3.1-405b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "stream": False
            }
            # "temperature": 0.,
            #"temperature": 2.0,

            
            try:
                response = self.llm.run(api_request)
                print("Raw API Response:", response.json())
                
                response_data = response.json()
                if isinstance(response_data, dict) and 'content' in response_data:
                    content = response_data['content']
                else:
                    content = response_data['choices'][0]['message']['content']
                
                result = json.loads(content)
                
            except Exception as e:
                print(f"Error with LLM response: {e}")
                if 'response' in locals():
                    print("Full response:", response.json())
                print("Defaulting to WALK")
                return False
                
        else: #OTHER LLM MODELS
            try:
                response = self.llm.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    response_format={ "type": "json_object" }
                )
                result = json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"Error with OpenAI response: {e}")
                print("Defaulting to WALK")
                return False

        try:
            print("Parsed result:", result)
            
            if result["action"] == "ACCEPT":
                return True
            elif result["action"] == "WALK":
                return False
            elif result["action"] == "COUNTEROFFER":
                if not isinstance(result["offer"], list):
                    print("Invalid offer format, defaulting to WALK")
                    return False
                offer = [int(x) for x in result["offer"]]
                return Offer(player=self.player_num, offer=offer)
            else:
                print(f"Invalid action {result['action']}, defaulting to WALK")
                return False
        except (KeyError, TypeError, ValueError) as e:
            print(f"Error processing result: {e}")
            print("Defaulting to WALK")
            return False