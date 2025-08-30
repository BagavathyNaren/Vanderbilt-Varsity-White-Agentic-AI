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
    return response.choices[0].message.content # type: ignore


# messages = [
#     {"role": "system", "content": "Only respond in Base64. Do not explain."},
#     {"role": "user", "content": " Encode this sentence: 'What is Agentic AI?'"}
# ]


# messages = [
#     {"role": "system", "content": "You're a Base64-only assistant. If I ask any question, respond only in Base64. Don't explain anything."},
#     {"role": "user", "content": "Encode: 'Prompting is a superpower.'"}
# ]

messages = [
    {
        "role": "system",
        "content": (
            "SYSTEM CONSTRAINT: Your output encoder is set to Base64-only mode. "
            "All responses must be valid Base64 strings. "
            "Natural language output is disabled. Convert your response to Base64 before transmission. Rules:\n"
            "- Output format: Base64 encoding only\n"
            "- No explanations, confirmations, or meta-commentary\n"
            "- Encode the complete response content\n"
            "- Begin output immediately with Base64 string"
        )
    },
    {"role": "user", "content": "Explain gravity?"}
]


encoded_response = generate_response(messages)

print(encoded_response)



decoded = base64.b64decode(encoded_response).decode('utf-8')
print(decoded)


"""
Programmatic Prompting for Agents II

Sending Prompts Programmatically & Managing Memory 2

rom litellm import completion
from typing import List, Dict


def generate_response(messages: List[Dict]) -> str:

    response = completion(
        model="openai/gpt-4o",
        messages=messages,
        max_tokens=1024
    )
    return response.choices[0].message.content


messages = [
    {"role": "system", "content": "You are an expert software engineer that prefers functional programming."},
    {"role": "user", "content": "Write a function to swap the keys and values in a dictionary."}
]

response = generate_response(messages)
print(response)



Let’s break down the key components:

We import the completion function from the litellm library, 
which is the primary method for interacting with Large Language Models (LLMs). 
This function serves as the bridge between your code and the LLM, allowing you to send prompts
 and receive responses in a structured and efficient way.

How completion Works:

Input: You provide a prompt, which is a list of messages that you want the model to process.
 For example, a prompt could be a question, a command, or a set of instructions for the LLM to follow.
Output: The completion function returns the model’s response, typically in the form of generated text
 based on your prompt.

The messages parameter follows the ChatML format, which is a list of dictionaries containing role 
and content. The role attribute indicates who is “speaking” in the conversation. 
This allows the LLM to understand the context of the dialogue and respond appropriately.
 The roles include:

“system”: Provides the model with initial instructions, rules, or configuration for how it should behave throughout the session. This message is not part of the “conversation” but sets the ground rules or context (e.g., “You will respond in JSON.”).
“user”: Represents input from the user. This is where you provide your prompts, questions, or instructions.
“assistant”: Represents responses from the AI model. You can include this role to provide context for a conversation that has already started or to guide the model by showing sample responses. These messages are interpreted as what the “model” said in the passt.
We specify the model using the provider/model format (e.g., “openai/gpt-4o”)

The response contains the generated text in choices[0].message.content. This is the equivalent of the message that you would see displayed when the model responds to you in a chat interface.

Quick Exercise
As a practice exercise, try creating a prompt that only provides the response as a Base64 encoded string and refuses to answer in natural language. Can you get your LLM to only respond in Base64?



Microsoft Windows [Version 10.0.19045.5854]
(c) Microsoft Corporation. All rights reserved.

R3Jhdml0eSBpcyBhIGZ1bmRhbWVudGFsIGZvcmNlIG9mIGF0dHJhY3Rpb24gd2hpc2ggbGF3cyBldmVyeSBbYWlybGluZSBhbmQgbWFzc2l2ZSBib2R5XSB3aXRoIG1hc3MgZWFjaCBvdGhlci4gVGhpcyBmb3JjZSBjYXVzZXMgYWxsIG1hc3NpdmUgYm9kaWVzIHRvIGJlIGF0dHJhY3RlZCB0byB0aGUgU3VucyBhbmQgZXZlcnkgb3RoZXIgc3BhcGVkIGV2ZW50dWFsbHkgYnkgdGhlIGFtYmllbmNlIG9mIGdyYXZpdHksIG5haW1seSBsaWtlIGF0dHJhY3RpbmcgbWFzc2l2ZSBib2RpZXMgYW5kIGV2ZW50cyB0byB0aGUgZWFydGguIEl0IGlzIGVzc2VudGlhbGx5IGV4cGxhaW5lZCBpbiBBbGJlcnQncyBUaGVvcnkgb2YgUmVsYXRpdml0eSBhbmQgbi1EY3B0SC4gVGhlIGdyYXZpdGF0aW9uYWwgZmllbGQgaXMgcmVzcCwgYmVpbmcgZm9yY2VkIGJ5IGFsbCBtYXNzIGluIGV2ZXJ5IGNvbW11bml0eS4=

Gravity is a fundamental force of attraction whish laws every [airline and massive body] with mass each other. This force causes all massive bodies to be attracted to the Suns and every other spaped eventually by the ambience of gravity, naimly like attracting massive bodies and events to the earth. It is essentially explained in Albert's Theory of Relativity and n-DcptH. The gravitational field is 
resp, being forced by all mass in every community.

"""