from agents.agent import Agent
from utils.offer import Offer
import json
from llamaapi import LlamaAPI
import openai
import os
import time
import anthropic
import google.generativeai as genai
from pydantic import BaseModel


class LLMAgent(Agent):
    def __init__(self, llm_type="llama", api_key=None, player_num=None):
        super().__init__(llm_type=llm_type, api_key=api_key)
        self.player_num = player_num
        self.llm_type = llm_type
        self.result = None
        self.action = None
        self.current_prompt = None #PROMPT SENT TO LLM
        self.current_response = None #RESPONSE FROM LLM (Text)
        
        if llm_type == "llama":
            if api_key is None:
                try:
                    # Get the directory of the current file
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    # Go up one level to project root and look for the API key file
                    key_path = os.path.join(os.path.dirname(current_dir), 'LLAMA_API_KEY.txt')
                    with open(key_path, 'r') as f:
                        api_key = f.read().strip()
                except FileNotFoundError:
                    raise ValueError("No API key provided and couldn't find LLAMA_API_KEY.txt")
            self.llm = LlamaAPI(api_key)
        elif llm_type == "openai":
            if api_key is None:
                try:
                    # Get the directory of the current file
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    # Go up one level to project root and look for the API key file
                    key_path = os.path.join(os.path.dirname(current_dir), 'OPEN_AI_API_KEY.txt')
                    with open(key_path, 'r') as f:
                        api_key = f.read().strip()
                except FileNotFoundError:
                    raise ValueError("No API key provided and couldn't find OPEN_AI_API_KEY.txt")
            self.llm = openai
            self.llm.api_key = api_key
        elif llm_type == "anthropic":
            if api_key is None:
                try:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    key_path = os.path.join(os.path.dirname(current_dir), 'ANTHROPIC_API_KEY.txt')
                    with open(key_path, 'r') as f:
                        
                        api_key = f.read().strip()
                except FileNotFoundError:
                    raise ValueError("No API key provided and couldn't find ANTHROPIC_API_KEY.txt")
            self.llm = anthropic.Anthropic(api_key=api_key)
        elif llm_type == "gemini":
            if api_key is None:
                try:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    key_path = os.path.join(os.path.dirname(current_dir), 'GEMINI_API_KEY.txt')
                    with open(key_path, 'r') as f:  
                        api_key = f.read().strip()
                except FileNotFoundError:
                    raise ValueError("No API key provided and couldn't find GEMINI_API_KEY.txt")
                
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel('gemini-2.0-flash-exp')
        else:
            raise ValueError(f"Invalid LLM type: {llm_type}")



    def safe_openai_request(model, messages, max_tokens):
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens
                )
                return response
            except openai.error.RateLimitError as e:
                wait_time = 1  
                if "Please try again in" in str(e):
                    import re
                    match = re.search(r"try again in (\d+)ms", str(e))
                    if match:
                        wait_time = int(match.group(1)) / 1000
                print(f"Rate limit reached. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
        
    def give_offer(self, prompt: str) -> Offer | bool:
        system_prompt = """You are an AI negotiator participating in a negotiation game. 

        You must respond in one of these formats ONLY:
        1. {"action": "ACCEPT"} - to accept the current offer
        2. {"action": "WALK"} - to walk away from negotiations
        3. {"action": "COUNTEROFFER", "offer": [n1, n2, ...]} - where n1, n2, ... are numbers representing your counteroffer. You recieve the items left over from your offer to the other player.

        Ensure your response is valid JSON and matches one of these exact formats."""
        
        if self.llm_type == "llama":
            api_request = {
                "model": "llama3.3-70b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
                #"temperature": 0.7,
                #"top_p": 0.9,
                #"top_k": 40
            }
            
            try:
                response = self.llm.run(api_request)
                print("Raw API Response:", response.text)
                self.current_response = response.text
                # Check if response is successful
                if response.status_code != 200:
                    print(f"API request failed with status code: {response.status_code}")
                    print(f"Error message: {response.text}")
                    raise Exception("API request failed")
                
                try:
                    response_json = json.loads(response.text)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse API response as JSON: {e}")
                    print(f"Raw response: {response.text}")
                    raise
                
                if 'choices' not in response_json:
                    print("Response missing 'choices' field")
                    print(f"Full response: {response_json}")
                    raise Exception("Invalid API response format")
                
                result_content = response_json['choices'][0]['message']['content']
                self.current_response = response_json['choices'][0]['message']['content']
                print("Extracted content:", result_content)
                
                # Find the JSON string at the end of the text and clean it
                json_start = result_content.rfind('{')
                if json_start == -1:
                    print("No JSON object found in response content")
                    print(f"Content: {result_content}")
                    raise Exception("No JSON found in response")
                    
                json_str = result_content[json_start:].strip()
                # Remove any trailing text after the JSON
                json_end = json_str.rfind('}')
                if json_end >= 0:
                    json_str = json_str[:json_end+1]
                
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse extracted JSON: {e}")
                    print(f"Extracted JSON string: {json_str}")
                    raise
                
                print("Parsed result:", result)
                self.result = result
                self.action = result["action"] 
                
            except Exception as e:
                print(f"Error with LLM response: {e}")
                print("Response details:")
                self.current_response = "Error with LLM response, did not receive a response."
                if 'response' in locals():
                    print(f"Status code: {response.status_code}")
                    print(f"Response text: {response.text}")
                    self.current_response = response.text
                print("Defaulting to WALK")
                result = {}
                result["action"] = "INVALID WALK"
                self.result = False
                self.action = "INVALID WALK"
                return False
                
        elif self.llm_type == "openai": 
            model = "gpt-4o" #gpt-4o #TODO: check OpenAI API type 
            try:
                response = {}
                if model == "gpt-4o":
                    response = openai.ChatCompletion.create(
                        model="gpt-4o", #o1-2024-12-17 #gpt-4o
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                    ],
                    #temperature=0.7,
                    #n=1,
                    #stop=None
                    )
                elif model == "gpt-o1":
                    response = openai.ChatCompletion.create(
                        model="o1-2024-12-17",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1024,
                        #response_format=js
                    )
                print("Raw API Response:", response)
                # Extract the action from the response text
                result_content = response.choices[0].message.content
                # Find the JSON string at the end of the text and clean it
                self.current_response = response.choices[0].message.content
                json_start = result_content.rfind('{')
                json_str = result_content[json_start:].strip()
                # Remove any trailing text after the JSON
                json_end = json_str.rfind('}') 
                if json_end >= 0:
                    json_str = json_str[:json_end+1]
                result = json.loads(json_str)
                print("Parsed result:", result)
                self.result = result
                self.action = result["action"]
            except Exception as e:
                print(f"Error with OpenAI response: {e}")
                if response is None:
                    self.current_response = "Error with OpenAI response, did not receive a response."
                else:
                    self.current_response = response
                print("Defaulting to WALK")
                result = {}
                result["action"] = "INVALID WALK"
                self.result = False
                self.action = "INVALID WALK"
                return False

        elif self.llm_type == "anthropic":
            try:
                response = self.llm.messages.create(
                    model="claude-3-5-sonnet-20241022", #claude-3-opus-20240229 , claude-3-5-sonnet-20241022
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1024
                )
                print("Raw API Response:", response)
                # Extract the action from the response text
                result_content = response.content[0].text
                self.current_response = response.content[0].text
                # Find the JSON string at the end of the text and clean it
                json_start = result_content.rfind('{')
                json_str = result_content[json_start:].strip()
                # Remove any trailing text after the JSON
                json_end = json_str.rfind('}') 
                if json_end >= 0:
                    json_str = json_str[:json_end+1]
                result = json.loads(json_str)
                print("Parsed result:", result)
                self.result = result
                self.action = result["action"]
            except Exception as e:
                print(f"Error with ANTHROPIC response: {e}")
                print("Defaulting to WALK")
                if response.content[0].text is not None:
                    self.current_response = response.content[0].text
                else:
                    self.current_response = "Error with ANTHROPIC response, did not receive a response."
                result = {}
                result["action"] = "INVALID WALK"
                self.result = False
                self.action = "INVALID WALK"
                return False
        elif self.llm_type == "gemini":
            try:
                response = self.llm.generate_content(prompt)
                print("Raw API Response:", response)
                result_content = response.candidates[0].content.parts[0].text
                self.current_response = response.candidates[0].content.parts[0].text
                # Find the JSON string at the end of the text and clean it
                json_start = result_content.rfind('{')
                json_str = result_content[json_start:].strip()
                # Remove any trailing text after the JSON
                json_end = json_str.rfind('}') 
                if json_end >= 0:
                    json_str = json_str[:json_end+1]
                result = json.loads(json_str)
                print("Parsed result:", result)
                self.result = result
                self.action = result["action"]
            except Exception as e:
                print(f"Error with Gemini response: {e}")
                print("Defaulting to WALK")
                self.current_response = "Error with Gemini response, did not receive a response."
                result = {}
                result["action"] = "INVALID WALK"
                self.result = False
                self.action = "INVALID WALK"
                return False
        else:
            raise ValueError(f"Invalid LLM type: {self.llm_type}")
        
        try:
            print("Parsed result:", result)
            
            if result["action"] == "ACCEPT":
                self.result = True
                self.action = "ACCEPT"
                return True
            elif result["action"] == "WALK":
                self.result = False
                self.action = "WALK"
                return False
            elif result["action"] == "COUNTEROFFER":
                if not isinstance(result["offer"], list):
                    print("Invalid offer format, defaulting to WALK")
                    self.result = False
                    result["action"] = "INVALID WALK"
                    self.action = "INVALID WALK"
                    return False
                offer = [int(x) for x in result["offer"]]
                self.result = Offer(player=self.player_num, offer=offer)
                self.action = "COUNTEROFFER"
                return Offer(player=self.player_num, offer=offer)
            else:
                print(f"Invalid action {result['action']}, defaulting to WALK")
                self.result = False
                self.action = "INVALID WALK"
                result["action"] = "INVALID WALK"
                return False
        except (KeyError, TypeError, ValueError) as e:
            print(f"Error processing result: {e}")
            print("Defaulting to WALK")
            self.result = False
            self.action = "INVALID WALK"
            result["action"] = "INVALID WALK"
            return False