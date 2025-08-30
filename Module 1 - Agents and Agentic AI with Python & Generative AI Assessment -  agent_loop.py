import json
import os
from typing import Dict, List
from litellm import completion
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


# Define agent rules
agent_rules = [{
    "role": "system",
    "content": """
You are an AI agent that can perform tasks by using available tools.

Available tools:
- list_files() -> List[str]: List all files in the current directory.
- read_file(file_name: str) -> str: Read the content of a file.
- terminate(message: str): End the agent loop and print a summary to the user.

If a user asks about files, list them before reading.

Every response MUST have an action.
Respond in this format:

```action
{
    "tool_name": "insert tool_name",
    "args": {...fill in any required arguments here...}
}
"""
}]

# Initialize memory to store interactions
memory: List[Dict] = []

# Define tool functions
def list_files() -> List[str]:
    return os.listdir('.')

def read_file(file_name: str) -> str:
    try:
        with open(file_name, 'r',encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{file_name}' not found."
    except Exception as e:
        return f"Error: {str(e)}"

def extract_markdown_block(response: str, block_type: str) -> str:
    start_marker = f"```{block_type}"
    end_marker = "```"
    start_idx = response.find(start_marker) + len(start_marker)
    end_idx = response.find(end_marker, start_idx)
    if start_idx == -1 or end_idx == -1:
        return response
    return response[start_idx:end_idx].strip()

def parse_action(response: str) -> Dict:
    try:
        response = extract_markdown_block(response, "action")
        response_json = json.loads(response)
        if "tool_name" in response_json and "args" in response_json:
            return response_json
        else:
            return {"tool_name": "error", "args": {"message": "You must respond with a JSON tool invocation."}}
    except json.JSONDecodeError:
        return {"tool_name": "error", "args": {"message": "Invalid JSON response. You must respond with a JSON tool invocation."}}

def generate_response(prompt: List[Dict]) -> str:
    response = completion(
        model="openai/gpt-4.1-mini",
        messages=prompt,
        temperature=0.7
    )
    return response.choices[0].message.content

def run_agent_loop(user_input: str, max_iterations: int = 10) -> None:
    global memory
    memory = [{"role": "user", "content": user_input}]
    iterations = 0

    while iterations < max_iterations:
        # 1. Construct prompt: Combine agent rules with memory
        prompt = agent_rules + memory

        # 2. Generate response from LLM
        print("Agent thinking...")
        response = generate_response(prompt)
        print(f"Agent response: {response}")

        # 3. Parse response to determine action
        action = parse_action(response)

        # 4. Execute action
        result = "Action executed"
        if action["tool_name"] == "list_files":
            result = {"result": list_files()}
        elif action["tool_name"] == "read_file":
            result = {"result": read_file(action["args"]["file_name"])}
        elif action["tool_name"] == "error":
            result = {"error": action["args"]["message"]}
        elif action["tool_name"] == "terminate":
            print(action["args"]["message"])
            break
        else:
            result = {"error": "Unknown action: " + action["tool_name"]}

        print(f"Action result: {result}")

        # 5. Update memory with response and results
        memory.extend([
            {"role": "assistant", "content": response},
            {"role": "user", "content": json.dumps(result)}
        ])

        # 6. Check termination condition
        if action["tool_name"] == "terminate":
            break

        iterations += 1

if __name__ == "__main__":
    # Example usage
    user_input = input("This is file read AI agent with agent loop using LiteLLM. Enter the input: ")
    run_agent_loop(user_input)


"""

user_input:

   What files are in this directory?
   Read the content of filename.txt
   Show me all files and then read the first one

"""