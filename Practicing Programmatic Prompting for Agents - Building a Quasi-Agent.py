"""

Building a Quasi-Agent:

For practice, we are going to write a quasi-agent that can write Python functions based on user requirements. 


It isn’t quite a real agent, it can’t react and adapt, but it can do something useful for us.

The quasi-agent will ask the user what they want code for, write the code for the function, add documentation,

and finally include test cases using the unittest framework. This exercise will help you understand how to
  
maintain context across multiple prompts and manage the information flow between the user and the LLM. 
    
    
It will also help you understand the pain of trying to parse and handle the output of an LLM that is not always consistent.

Practice Exercise

This exercise will allow you to practice programmatically sending prompts to an LLM and managing memory.

For this exercise, you should write a program that uses sequential prompts to generate any Python function

based on user input. The program should:

First Prompt:

Ask the user what function they want to create
Ask the LLM to write a basic Python function based on the user’s description
Store the response for use in subsequent prompts
Parse the response to separate the code from the commentary by the LLM
Second Prompt:

Pass the code generated from the first prompt
Ask the LLM to add comprehensive documentation including:
Function description
Parameter descriptions
Return value description
Example usage
Edge cases
Third Prompt:

Pass the documented code generated from the second prompt
Ask the LLM to add test cases using Python’s unittest framework
Tests should cover:
Basic functionality
Edge cases
Error cases
Various input scenarios
Requirements:

Use the LiteLLM library
Maintain conversation context between prompts
Print each step of the development process
Save the final version to a Python file
If you want to practice further, try using the system message to force the LLM to always output code that has a specific style or uses particular libraries.

"""


from dotenv import load_dotenv
import os
from litellm import completion
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# Unified function to send messages using LiteLLM
def ask_llm(prompt, temperature=0.3):
    response = completion(
        model="openai/gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response['choices'][0]['message']['content'].strip()

# Step 1: Generate basic function from user input
def generate_basic_function():
    user_description = input("What Python function would you like to create? ")
    prompt = f"Write a simple Python function based on this description:\n'{user_description}'. Keep it minimal."
    response = ask_llm(prompt)

    print("\n=== Basic Function (Raw LLM Output) ===")
    print(response)

    # Extract code using regex
    code_blocks = re.findall(r"```python(.*?)```", response, re.DOTALL)
    code = code_blocks[0].strip() if code_blocks else response.strip()
    return code

# Step 2: Add full documentation to the code
def add_documentation(code):
    prompt = (
        f"Here is a Python function:\n\n{code}\n\n"
        "Please add full documentation including:\n"
        "- A docstring with function description\n"
        "- Descriptions for all parameters\n"
        "- Return value description\n"
        "- Example usage\n"
        "- Possible edge cases"
    )
    response = ask_llm(prompt)
    print("\n=== Documented Function ===")
    print(response)
    return response

# Step 3: Add unit tests
def add_tests(documented_code):
    prompt = (
        f"Here is a documented Python function:\n\n{documented_code}\n\n"
        "Please write unit tests using Python’s unittest module. "
        "Include tests for basic functionality, edge cases, error cases, and different input types."
    )
    response = ask_llm(prompt)
    print("\n=== Unit Tests ===")
    print(response)
    return response

# Run the full workflow
if __name__ == "__main__":
    basic_code = generate_basic_function()
    documented_code = add_documentation(basic_code)
    test_code = add_tests(documented_code)