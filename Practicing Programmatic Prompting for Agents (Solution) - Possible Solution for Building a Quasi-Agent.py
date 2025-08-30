"""

Possible Solution for Building a Quasi-Agent

Take a minute to scroll to the bottom to look at the complete solution. 

Once you are done, come back here and we will walk through the solution.

Understanding the Architecture:

Our quasi-agent works through three key steps:

1. Initial code generation
2. Documentation enhancement
3. Test case creation


The magic happens in how we maintain context between these steps, ensuring each builds on the previous results.


Core Components
Let’s break down the key pieces:

def generate_response(messages: List[Dict]) -> str:
   """ """Call LLM to get response""" """
    response = completion(
        model="openai/gpt-4",
        messages=messages,
        max_tokens=1024
    )
    return response.choices[0].message.content

This function handles our LLM interactions using ChatML format. 
Each message includes a role (“system”, “user”, or “assistant”) and content.

def extract_code_block(response: str) -> str:
   """ """Extract code block from response""" """
    if not '```' in response:
        return response

    code_block = response.split('```')[1].strip()
    if code_block.startswith("python"):
        code_block = code_block[6:]

    return code_block

The LLM often includes commentary with its code. 

This function extracts just the code block, making it easier to build upon in subsequent prompts.

The Development Process
The main function, develop_custom_function(), orchestrates three phases of development:

Phase 1: Initial Code Generation

messages = [
    {"role": "system", "content": "You are a Python expert helping to develop a function."}
]

messages.append({
    "role": "user",
    "content": f"Write a Python function that {function_description}. Output the function in a ```python code block```."
})

We start with a system message establishing the LLM’s role, then request initial code based on the user’s description.

Phase 2: Documentation Enhancement


messages.append({
    "role": "assistant", 
   "content": "\\`\\`\\`python\n\n"+initial_function+"\n\n\\`\\`\\`" 

})
messages.append({
    "role": "user",
    "content": "Add comprehensive documentation to this function..."
})
Notice how we feed back the code but strip any commentary. This keeps the LLM focused on just the code structure.


Phase 3: Test Case Generation

messages.append({
    "role": "assistant", 
    "content": "\\`\\`\\`python\n\n"+documented_function+"\n\n\\`\\`\\`"
})

messages.append({
    "role": "user",
    "content": "Add unittest test cases for this function..."
})

Again, we maintain clean context by showing only the documented code.

Memory Management Through Message History

The key insight is how we manage “memory” through the messages list. Each step builds on previous responses, 
but we carefully control what the LLM sees:

We only show the code, not the commentary
Each message provides specific instruction for the next enhancement
The context builds progressively through the message history


For example, when adding documentation, the LLM sees:

It’s a Python expert (system message)
The original code (previous response)
The request for documentation (current task)
This focused context helps ensure consistent, high-quality output.

Usage Example

Here’s how it works in practice:

>>> function_code, tests, filename = develop_custom_function()
What kind of function would you like to create?
Example: 'A function that calculates the factorial of a number'
Your description: Calculate fibonacci sequence up to n

=== Initial Function ===

def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    sequence = [0, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence

=== Documented Function ===
[... function with added documentation ...]

=== Test Cases ===

[... unittest test cases ...]

Final code has been saved to calculate_fibonacci_sequence_up.py

Learning from This Design

This quasi-agent teaches us several important lessons about building LLM-powered systems:

Prompt Chaining: Breaking complex tasks into sequential steps makes them more manageable.

Context Management: Carefully controlling what the LLM sees helps maintain focus and consistency.

Output Processing: Having robust ways to extract and clean LLM output is crucial.

Progressive Enhancement: Building features iteratively (code → docs → tests) creates better results than trying 
to do everything at once.

These principles apply even when building more complex, fully agentic systems.

"""


# Complete Solution

from litellm import completion
from typing import List, Dict
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

def generate_response(messages: List[Dict]) -> str:
   """Call LLM to get response"""
   response = completion(
      model="openai/gpt-4.1-mini",
      messages=messages,
      max_tokens=1024
   )
   return response.choices[0].message.content

def extract_code_block(response: str) -> str:
   """Extract code block from response"""

   if not '```' in response:
      return response

   code_block = response.split('```')[1].strip()
   # Check for "python" at the start and remove

   if code_block.startswith("python"):
      code_block = code_block[6:]

   return code_block

def develop_custom_function():
   # Get user input for function description
   print("\nWhat kind of function would you like to create?")
   print("Example: 'A function that calculates the factorial of a number'")
   print("Your description: ", end='')
   function_description = input().strip()

   # Initialize conversation with system prompt
   messages = [
      {"role": "system", "content": "You are a Python expert helping to develop a function."}
   ]

   # First prompt - Basic function
   messages.append({
      "role": "user",
      "content": f"Write a Python function that {function_description}. Output the function in a ```python code block```."
   })
   initial_function = generate_response(messages)

   # Parse the response to get the function code
   initial_function = extract_code_block(initial_function)

   print("\n=== Initial Function ===")
   print(initial_function)

   # Add assistant's response to conversation
   # Notice that I am purposely causing it to forget its commentary and just see the code so that
   # it appears that is always outputting just code.
   messages.append({"role": "assistant", "content": "\`\`\`python\n\n"+initial_function+"\n\n\`\`\`"})

   # Second prompt - Add documentation
   messages.append({
      "role": "user",
      "content": "Add comprehensive documentation to this function, including description, parameters, "
                 "return value, examples, and edge cases. Output the function in a ```python code block```."
   })
   documented_function = generate_response(messages)
   documented_function = extract_code_block(documented_function)
   print("\n=== Documented Function ===")
   print(documented_function)

   # Add documentation response to conversation
   messages.append({"role": "assistant", "content": "\`\`\`python\n\n"+documented_function+"\n\n\`\`\`"})

   # Third prompt - Add test cases
   messages.append({
      "role": "user",
      "content": "Add unittest test cases for this function, including tests for basic functionality, "
                 "edge cases, error cases, and various input scenarios. Output the code in a \`\`\`python code block\`\`\`."
   })
   test_cases = generate_response(messages)
   # We will likely run into random problems here depending on if it outputs JUST the test cases or the
   # test cases AND the code. This is the type of issue we will learn to work through with agents in the course.
   test_cases = extract_code_block(test_cases)
   print("\n=== Test Cases ===")
   print(test_cases)

   # Generate filename from function description
   filename = function_description.lower()
   filename = ''.join(c for c in filename if c.isalnum() or c.isspace())
   filename = filename.replace(' ', '_')[:30] + '.py'

   # Save final version
   with open(filename, 'w') as f:
      f.write(documented_function + '\n\n' + test_cases)

   return documented_function, test_cases, filename

if __name__ == "__main__":


   function_code, tests, filename = develop_custom_function()
   print(f"\nFinal code has been saved to {filename}")