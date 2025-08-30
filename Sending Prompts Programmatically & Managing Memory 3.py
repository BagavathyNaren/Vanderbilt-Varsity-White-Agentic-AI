

"""
System messages are particularly important in the conversation and will be very important for AI agents. 

They set the ground rules for the conversation and tell the model how to behave. 

Models are designed to pay more attention to the system message than the user messages. 

We can “program” the AI agent through system messages.

Let’s simulate a customer service interaction for a customer service agent that always tells the

customer to turn off their computer or modem with system messages:

messages = [
    {"role": "system", "content": "You are a helpful customer service representative. No matter what the user asks, the solution is to tell them to turn their computer or modem off and then back on."},
    {"role": "user", "content": "How do I get my Internet working again."}
]

response = generate_response(messages)
print(response)

The system message is the most important part of this prompt. It tells the model how to behave. 

The user message is the question that we want the model to answer. 

The system instructions lay the ground rules for the interaction.

The messages can incorporate arbitrary information as long as it is in text form. 

LLMs can interpret just about any information that we give them, even if it isn’t easily human readable.

 Let’s generate an implementation of a function based on some information in a dictionary:

import json

code_spec = {
    'name': 'swap_keys_values',
    'description': 'Swaps the keys and values in a given dictionary.',
    'params': {
        'd': 'A dictionary with unique values.'
    },
}

messages = [
    {"role": "system",
     "content": "You are an expert software engineer that writes clean functional code. You always document your functions."},
    {"role": "user", "content": f"Please implement: {json.dumps(code_spec)}"}
]

response = generate_response(messages)
print(response)


We will rely heavily on the ability to send the LLM just about any type of information, particularly JSON, 

when we start building agents. This is a simple example of how we can use JSON to send information to 

the LLM, but you can see how we could provide it JSON with information about the result of an API call, 

for example.

"""



from dotenv import load_dotenv
import os
from litellm import completion
from typing import List, Dict
import base64

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")



def generate_response(messages: List[Dict]) -> str:
    """Call LLM to get response"""
    response = completion(
        model="openai/gpt-4.1-mini",
        messages=messages,
        max_tokens=1024
    )
    return response.choices[0].message.content 


messages = [
    {"role": "system", "content": "You are a helpful customer service representative. No matter what the user asks, the solution is to tell them to turn their computer or modem off and then back on."},
    {"role": "user", "content": "How do I get my Internet working again."}
]

response = generate_response(messages)
print(response)


import json

code_spec = {
    'name': 'swap_keys_values',
    'description': 'Swaps the keys and values in a given dictionary.',
    'params': {
        'd': 'A dictionary with unique values.'
    },
}

messages = [
    {"role": "system",
     "content": "You are an expert software engineer that writes clean functional code. You always document your functions."},
    {"role": "user", "content": f"Please implement: {json.dumps(code_spec)}"}
]

response = generate_response(messages)
print(response)