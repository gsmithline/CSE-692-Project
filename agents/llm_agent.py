import json
import os
import time

import anthropic 
import google.generativeai as genai
import openai
from llamaapi import LlamaAPI
from pydantic import BaseModel

from agents.agent import Agent
from utils.offer import Offer


class LLMAgent(Agent):
    """
    A negotiator agent that interfaces with various large language models
    (LLMs) to propose or respond to offers. It handles multiple providers:
    llama, openai, anthropic, and gemini. 
    """

    def __init__(self, llm_type="llama", model="llama3.3-70b", api_key=None, player_num=None):
        """
        Initialize the LLMAgent with the specified LLM type and optional API key.
        If the API key is not provided, it attempts to read it from a file
        named '<LLM>_API_KEY.txt' in the project's root directory.
        
        :param llm_type: The type of LLM backend (e.g., 'llama', 'openai').
        :param api_key: The API key for the LLM. If None, attempts to load from file.
        :param player_num: Identifier for the player using this agent.
        """
        super().__init__(llm_type=llm_type, api_key=api_key)
        self.player_num = player_num
        self.llm_type = llm_type
        self.result = None
        self.action = None
        self.current_prompt = None  # Prompt sent to LLM
        self.current_response = None  # Response text from LLM
        self.model = model
        # Determine which LLM to use and load or configure accordingly.
        if "llama" in llm_type:
            if api_key is None:
                try:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    key_path = os.path.join(os.path.dirname(current_dir), 'LLAMA_API_KEY.txt')
                    with open(key_path, 'r') as f:
                        api_key = f.read().strip()
                except FileNotFoundError:
                    raise ValueError("No API key provided and couldn't find LLAMA_API_KEY.txt")
            self.llm = LlamaAPI(api_key)

        elif "openai" in llm_type:
            if api_key is None:
                try:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    key_path = os.path.join(os.path.dirname(current_dir), 'OPEN_AI_API_KEY.txt')
                    with open(key_path, 'r') as f:
                        api_key = f.read().strip()
                except FileNotFoundError:
                    raise ValueError("No API key provided and couldn't find OPEN_AI_API_KEY.txt")
            self.llm = openai
            self.llm.api_key = api_key

        elif "anthropic" in llm_type:
            if api_key is None:
                try:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    key_path = os.path.join(os.path.dirname(current_dir), 'ANTHROPIC_API_KEY.txt')
                    with open(key_path, 'r') as f:
                        api_key = f.read().strip()
                except FileNotFoundError:
                    raise ValueError("No API key provided and couldn't find ANTHROPIC_API_KEY.txt")
            self.llm = anthropic.Anthropic(api_key=api_key)

        elif "gemini" in llm_type:
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

    def give_offer(self, prompt: str) -> Offer | bool:
        """
        Given a prompt describing the negotiation context, request a proposal 
        from the LLM. The LLM must respond in one of three JSON formats:

          1. {"action": "ACCEPT"}
          2. {"action": "WALK"}
          3. {"action": "COUNTEROFFER", "offer": [n1, n2, ...]}

        Return:
            - True if "ACCEPT" was requested.
            - False if "WALK" was requested or the LLM response is invalid.
            - An Offer object if "COUNTEROFFER" was requested.
        """
        system_prompt = (
            "You are an AI negotiator participating in a negotiation game.\n\n"
            "You must respond in one of these formats ONLY:\n"
            "1. {\"action\": \"ACCEPT\"}\n"
            "2. {\"action\": \"WALK\"}\n"
            "3. {\"action\": \"COUNTEROFFER\", \"offer\": [n1, n2, ...]}\n\n"
            "Ensure your response is valid JSON and matches one of these exact formats."
        )

        # LLAMA branch
        if self.llm_type == "llama":
            #default to llama3.3-70b
            api_request = {
                "model": self.model, 
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
            }

            try:
                response = self.llm.run(api_request)
                print("Raw API Response:", response.text)
                self.current_response = response.text

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
                self.current_response = result_content
                print("Extracted content:", result_content)

                # Find the JSON string at the end of the text and clean it
                json_start = result_content.rfind('{')
                if json_start == -1:
                    print("No JSON object found in response content")
                    print(f"Content: {result_content}")
                    raise Exception("No JSON found in response")

                json_str = result_content[json_start:].strip()
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
                result = {"action": "INVALID WALK"}
                self.result = False
                self.action = "INVALID WALK"
                return False

        # OpenAI branch
        elif "openai" in self.llm_type:
            model = self.llm_type
            try:
                response = {}
                if "4o" in model:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                    )
                elif model == "gpt-o1":
                    response = openai.ChatCompletion.create(
                        model="o1-2024-12-17",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1024
                    )
                elif "o3_mini" in model:
                    response = openai.ChatCompletion.create(
                        model="o3-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                    )

                print("Raw API Response:", response)
                result_content = response.choices[0].message.content
                self.current_response = result_content

                # Find the JSON string at the end of the text and clean it
                json_start = result_content.rfind('{')
                json_str = result_content[json_start:].strip()
                json_end = json_str.rfind('}')
                if json_end >= 0:
                    json_str = json_str[:json_end+1]

                result = json.loads(json_str)
                print("Parsed result:", result)
                self.result = result
                self.action = result["action"]

            except Exception as e:
                print(f"Error with OpenAI response: {e}")
                if not response:
                    self.current_response = "Error with OpenAI response, did not receive a response."
                else:
                    self.current_response = response
                print("Defaulting to WALK")
                result = {"action": "INVALID WALK"}
                self.result = False
                self.action = "INVALID WALK"
                return False

        # Anthropic branch
        elif self.llm_type == "anthropic":
            if "3.5" in self.model:
                model = "claude-3-5-sonnet-20241022"
            elif "3.0" in self.model:
                model = "claude-3-opus-20240229"
            elif "3.0" in self.model:
                model = "claude-3-opus-20240229"
            else:
                raise ValueError(f"Invalid model: {self.model}")
            self.model = model
            try:
                response = self.llm.messages.create(
                    model=model, 
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024
                    )
                print("Raw API Response:", response)

                result_content = response.content[0].text
                self.current_response = result_content

                json_start = result_content.rfind('{')
                json_str = result_content[json_start:].strip()
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
                if response.content[0].text:
                    self.current_response = response.content[0].text
                else:
                    self.current_response = "Error with ANTHROPIC response, did not receive a response."
                result = {"action": "INVALID WALK"}
                self.result = False
                self.action = "INVALID WALK"
                return False

        elif self.llm_type == "gemini":
            try:
                response = self.llm.generate_content(prompt)
                print("Raw API Response:", response)

                result_content = response.candidates[0].content.parts[0].text
                self.current_response = result_content

                json_start = result_content.rfind('{')
                json_str = result_content[json_start:].strip()
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
                result = {"action": "INVALID WALK"}
                self.result = False
                self.action = "INVALID WALK"
                return False

        else:
            raise ValueError(f"Invalid LLM type: {self.llm_type}")

        # Process and return the parsed result
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
                return False

        except (KeyError, TypeError, ValueError) as e:
            print(f"Error processing result: {e}")
            print("Defaulting to WALK")
            self.result = False
            self.action = "INVALID WALK"
            return False
