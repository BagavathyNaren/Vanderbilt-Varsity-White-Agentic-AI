import json
import os
import re
from typing import List
from litellm import completion
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# ---------------------- Utility Functions ----------------------

def extract_directory_path(user_input: str) -> str:
    """Extracts directory path from angle brackets <> in the user prompt."""
    match = re.search(r"<(.+?)>", user_input)
    if match:
        path = match.group(1).strip('"')
        if os.path.isdir(path):
            print(f"[DEBUG] Directory found: {path}")
            return path
        else:
            print(f"[ERROR] Not a valid directory: {path}")
            return None
    else:
        print("[ERROR] No directory path found in angular brackets <>.")
        return None


def list_files(directory: str) -> List[str]:
    """List files in the specified directory."""
    try:
        return os.listdir(directory)
    except Exception as e:
        return [f"Error: {str(e)}"]


def read_file(file_name: str, directory: str) -> str:
    """Read a file's contents from a specific directory."""
    file_path = os.path.join(directory, file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: {file_path} not found."
    except Exception as e:
        return f"Error: {str(e)}"


def terminate(message: str) -> None:
    """Terminate the agent loop and provide a summary message."""
    print(f"Termination message: {message}")


# ---------------------- Agent Setup ----------------------

# Get user input and extract directory
user_task = input("What would you like me to do? ")
target_directory = extract_directory_path(user_task)

# Exit if directory is invalid
if not target_directory:
    print("Exiting: Invalid or missing directory path.")
    exit(1)

# Tool function wrappers with bound directory
def list_files_with_dir() -> List[str]:
    print(f"[DEBUG] Listing files in: {target_directory}")
    return list_files(target_directory)

def read_file_with_dir(file_name: str) -> str:
    print(f"[DEBUG] Reading file '{file_name}' from: {target_directory}")
    return read_file(file_name, target_directory)

# Tool functions
tool_functions = {
    "list_files": list_files_with_dir,
    "read_file": read_file_with_dir,
    "terminate": terminate
}

# Tool metadata
tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": f"Returns a list of files in the directory: {target_directory}",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": f"Reads the content of a specified file in the directory: {target_directory}",
            "parameters": {
                "type": "object",
                "properties": {"file_name": {"type": "string"}},
                "required": ["file_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "terminate",
            "description": "Terminates the conversation. Prints the message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                },
                "required": ["message"]
            }
        }
    }
]

# Agent behavior
agent_rules = [{
    "role": "system",
    "content": """
You are an AI agent that can perform tasks by using available tools.

If a user asks about files, documents, or content, first list the files before reading them.

When you are done, terminate the conversation by using the "terminate" tool and I will provide the results to the user.
"""
}]

# ---------------------- Agent Loop ----------------------

memory = [{"role": "user", "content": user_task}]
iterations = 0
max_iterations = 10

while iterations < max_iterations:
    messages = agent_rules + memory

    response = completion(
        model="openai/gpt-4.1-mini",
        messages=messages,
        tools=tools,
        max_tokens=1024
    )

    if response.choices[0].message.tool_calls:
        tool = response.choices[0].message.tool_calls[0]
        tool_name = tool.function.name
        tool_args = json.loads(tool.function.arguments)

        action = {
            "tool_name": tool_name,
            "args": tool_args
        }

        if tool_name == "terminate":
            print(f"Termination message: {tool_args['message']}")
            break
        elif tool_name in tool_functions:
            try:
                result = {"result": tool_functions[tool_name](**tool_args)}
            except Exception as e:
                result = {"error": f"Error executing {tool_name}: {str(e)}"}
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        print(f"Executing: {tool_name} with args {tool_args}")
        print(f"Result: {result}")
        memory.extend([
            {"role": "assistant", "content": json.dumps(action)},
            {"role": "user", "content": json.dumps(result)}
        ])
        iterations += 1
    else:
        print("Response:", response.choices[0].message.content)
        break