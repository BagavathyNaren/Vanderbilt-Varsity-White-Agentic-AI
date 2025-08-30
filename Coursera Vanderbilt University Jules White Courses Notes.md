Agent Tool Design Best Practices

Tool Design and Naming Best Practices for AI Agents
Designing Effective Tools for AI Agents
When designing tools for an AI agent, the goal is to provide a limited, well-defined set of functions that are as specific as possible to the agent’s intended task. Well-designed tools reduce ambiguity, improve reliability, and help the AI execute actions correctly without misinterpretation.

Why Tool Design Matters
Sometimes, if tools are too generic, such as a single list_files or read_file function, the AI might struggle to use them correctly. For instance, an agent might attempt to read a file but specify the wrong directory, leading to errors. Instead, tools should be structured to enforce correctness while minimizing the agent’s margin for error.

Agents can use generic tools as well, but more specialized tools are easier to manage and less prone to misuse by the agent. There is a trade-off between the specificity of tools and the flexibility they provide. More specific tools also limit reuse, so finding the right balance is crucial. When building an agent inititally, err on the side of specificity.

Let’s assume that rather than a generic file agent, we are writing an agent that works with Python code and documentation. Instead of defining broad functions like:

list_files(directory: str) → Returns all files in a specified directory.
read_file(file_path: str) → Reads a file from an arbitrary path.
write_file(file_path: str, content: str) → Writes to an arbitrary file.
We should define task-specific functions like:

list_python_files() → Returns Python files only from the src/ directory.
read_python_file(file_name: str) → Reads a Python file only from the src/ directory.
write_documentation(file_name: str, content: str) → Writes documentation only to the docs/ directory.
In the context of the more limited scope of Python documentation, the constraints reduce the chances of incorrect agent behavior while making it clear what each tool does.

Example: Reading a Python File
{
  "tool_name": "read_python_file",
  "description": "Reads the content of a Python file from the src/ directory.",
  "parameters": {
    "type": "object",
    "properties": {
      "file_name": { "type": "string" }
    },
    "required": ["file_name"]
  }
}
Example: Writing Documentation
{
  "tool_name": "write_documentation",
  "description": "Writes a documentation file to the docs/ directory.",
  "parameters": {
    "type": "object",
    "properties": {
      "file_name": { "type": "string" },
      "content": { "type": "string" }
    },
    "required": ["file_name", "content"]
  }
}
Step 2: Naming Matters – Best Practices for Tool Naming
Naming plays a crucial role in AI comprehension. If we name a tool proc_handler, the AI might struggle to infer its purpose. Instead, naming it process_file provides better clarity.

Example: Naming Comparison
Poor Name	Better Name
list_pf	list_python_files
rd_f	read_python_file
wrt_doc	write_documentation
Even with well-named tools, we still need structured descriptions for disambiguation, especially in specialized domains.

Step 3: Robust Error Handling in Tools
Each tool should be designed to handle errors gracefully and provide rich error messages back to the agent. This helps prevent failures and enables the agent to adjust its actions dynamically when unexpected issues occur.

Example: Improving read_python_file with Error Handling
import os

def read_python_file(file_name):
    """Reads a Python file from the src/ directory with error handling."""
    file_path = os.path.join("src", file_name)
    
    if not file_name.endswith(".py"):
        return {"error": "Invalid file type. Only Python files can be read."}
    
    if not os.path.exists(file_path):
        return {"error": f"File '{file_name}' does not exist in the src/ directory."}
    
    with open(file_path, "r") as f:
        return {"content": f.read()}
This version ensures:

Only Python files are read.
Non-existent files return an informative error.
All responses are structured for easy agent parsing.
Instructions in Error Messages
When we know that certain tools will always be used together or that there is a right way to handle a particular error, we can provide that information back to the agent in the error message. For example, in our read_python_file function, if the file does not exist, we can suggest that the agent call the list_python_files function to get an accurate list of file names.

def read_python_file(file_name):
    """Reads a Python file from the src/ directory with error handling."""
    file_path = os.path.join("src", file_name)
    
    if not file_name.endswith(".py"):
        return {"error": "Invalid file type. Only Python files can be read. Call the list_python_files function to get a list of valid files."}
  ...
We could put this information into the original agent rules, but then the agent would have to remember it. Further, adding it there creates a more complicated set of rules for the agent to remember and may have unexpected side effects on how it handles errors in other tools. By injecting the instruction here, we can ensure that the agent has the information it needs to handle this error without having to remember it in its rules. It also gets the instruction “just in time” when it needs it.

Conclusion
When integrating AI into real-world environments, tool descriptions must be explicit, structured, and informative. By following these principles:

Use descriptive names.
Provide structured metadata.
Leverage JSON Schema for parameters.
Ensure AI has contextual understanding.
Include robust error handling.
Provide informative error messages.
Inject instructions into error messages.
This approach ensures that AI agents can interact with their environments effectively while minimizing incorrect or ambiguous tool usage. In the next tutorial, we will explore dynamic tool registration using decorators, further improving flexibility and maintainability in AI agent design.

Module 2 Using LLM Function Calling for AI-Agent Interaction

Using LLM Function Calling for AI-Agent Interaction

Step 3: Using LLM Function Calling for Structured Execution
One of the most challenging aspects of integrating AI agents with tool execution is ensuring that the model consistently produces structured output that can be parsed correctly. Traditionally, developers would attempt to engineer prompts to make the model output well-formed JSON, but this approach is unreliable—models can introduce variations, omit required fields, or output unstructured text that breaks parsing logic.

To solve this, most LLMs offer function calling APIs that guarantee structured execution. Instead of treating function execution as a free-form text generation task, function calling APIs allow us to explicitly define the tools available to the model using JSON Schema. The model then decides when and how to call these functions, ensuring structured and predictable responses.

How Function Calling Simplifies AI-Agent Interfaces
When using function calling, the model returns either:

A function call that includes the tool name and arguments as structured JSON.
A standard text response if the model decides a function is unnecessary.
This approach removes the need for manual prompt engineering to enforce structured output and allows the agent to focus on decision-making rather than syntax compliance.

Example: Executing Function Calls
Below is a typical way to invoke function calling using OpenAI’s API:

import json
import os
from typing import List

from litellm import completion

def list_files() -> List[str]:
    """List files in the current directory."""
    return os.listdir(".")

def read_file(file_name: str) -> str:
    """Read a file's contents."""
    try:
        with open(file_name, "r") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: {file_name} not found."
    except Exception as e:
        return f"Error: {str(e)}"


tool_functions = {
    "list_files": list_files,
    "read_file": read_file
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Returns a list of files in the directory.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the content of a specified file in the directory.",
            "parameters": {
                "type": "object",
                "properties": {"file_name": {"type": "string"}},
                "required": ["file_name"]
            }
        }
    }
]

# Our rules are simplified since we don't have to worry about getting a specific output format
agent_rules = [{
    "role": "system",
    "content": """
You are an AI agent that can perform tasks by using available tools. 

If a user asks about files, documents, or content, first list the files before reading them.
"""
}]

user_task = input("What would you like me to do? ")

memory = [{"role": "user", "content": user_task}]

messages = agent_rules + memory

response = completion(
    model="openai/gpt-4o",
    messages=messages,
    tools=tools,
    max_tokens=1024
)

# Extract the tool call from the response, note we don't have to parse now!
tool = response.choices[0].message.tool_calls[0]
tool_name = tool.function.name
tool_args = json.loads(tool.function.arguments)
result = tool_functions[tool_name](**tool_args)

print(f"Tool Name: {tool_name}")
print(f"Tool Arguments: {tool_args}")
print(f"Result: {result}")

Breaking Down Function Calling Step by Step
Let’s examine how function calling works in detail:

1. Define the Tool Functions
def list_files() -> List[str]:
    """List files in the current directory."""
    return os.listdir(".")

def read_file(file_name: str) -> str:
    """Read a file's contents."""
    try:
        with open(file_name, "r") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: {file_name} not found."
    except Exception as e:
        return f"Error: {str(e)}"
First, we define the actual Python functions that will be executed. These contain the business logic for each tool and handle the actual operations the AI agent can perform.

2. Create a Function Registry
tool_functions = {
    "list_files": list_files,
    "read_file": read_file
}
We maintain a dictionary that maps function names to their corresponding Python implementations. This registry allows us to easily look up and execute the appropriate function when the model calls it.

3. Define Tool Specifications Using JSON Schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Returns a list of files in the directory.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the content of a specified file in the directory.",
            "parameters": {
                "type": "object",
                "properties": {"file_name": {"type": "string"}},
                "required": ["file_name"]
            }
        }
    }
]
This is where we describe our tools to the model. Each tool specification includes:

A name that matches a key in our tool_functions dictionary
A description that helps the model understand when to use this tool
Parameters defined using JSON Schema, specifying the expected input format
Note how the list_files function takes no parameters (empty “properties” object), while read_file requires a “file_name” string parameter. The model will use these specifications to generate properly structured calls.

4. Set Up the Agent’s Instructions
agent_rules = [{
    "role": "system",
    "content": """
You are an AI agent that can perform tasks by using available tools. 

If a user asks about files, documents, or content, first list the files before reading them.
"""
}]
The system message provides guidance on how the agent should behave. With function calling, we don’t need to instruct the model on how to format its outputs - we only need to focus on the decision-making logic.

5. Prepare the Conversation Context
user_task = input("What would you like me to do? ")
memory = [{"role": "user", "content": user_task}]
messages = agent_rules + memory
We combine the system instructions with the user’s input to create the conversation context.

6. Make the API Call with Function Definitions
response = completion(
    model="openai/gpt-4o",
    messages=messages,
    tools=tools,
    max_tokens=1024
)
The critical difference here is the inclusion of the tools parameter, which tells the model what functions it can call. This is what activates the function calling mechanism.

7. Process the Structured Response
tool = response.choices[0].message.tool_calls[0]
tool_name = tool.function.name
tool_args = json.loads(tool.function.arguments)
When using function calling, the response comes back with a dedicated tool_calls array rather than free-text output. This ensures that:

The function name is properly identified
The arguments are correctly formatted as valid JSON
We don’t need to parse or extract from unstructured text
8. Execute the Function with the Provided Arguments
result = tool_functions[tool_name](**tool_args)
Finally, we look up the appropriate function in our registry and call it with the arguments the model provided. The **tool_args syntax unpacks the JSON object into keyword arguments.

Key Benefits of Function Calling APIs

Eliminates prompt engineering for structured responses – No need to force the model to output JSON manually.

Uses standardized JSON Schema – The same format used in API documentation applies seamlessly to AI interactions.

Allows mixed text and tool execution – The model can decide whether a tool is necessary or provide a natural response.

Simplifies parsing logic – Instead of handling inconsistent outputs, developers only check for tool_calls in the response.

Guarantees syntactically correct arguments – The model automatically ensures arguments match the expected parameter format.

Conclusion

Function calling APIs significantly improve the reliability of AI-agent interactions by enforcing structured execution. By defining tools with JSON Schema and letting the model determine when to use them, we build a more predictable and maintainable AI environment interface. In the next section, we will explore how to register these tools dynamically using decorators to further streamline agent development.


Try Out LLM Function Calling

When you run the second block in the notebook, the agent will prompt you for what action to take. You can say something like "tell me the files in the current directory" and it will make the appropriate tool choice. Experiment with the agent to see what it can and cannot do. 


Look for this beneath the code when you run the second block. Type in a task and hit enter. Remember, it only has tools for listing and reading files.


https://colab.research.google.com/drive/1pLBChdZirjle6v3-iqMTIu0YJmk7ZPD6?usp=sharing



                                      An Agent Loop with Function Calling


Simplifying the AI Agent Loop with Function Calling

Function Calling and the Simplified Agent Loop

Now that we have explored function calling, we can significantly simplify our AI agent loop. Traditionally, parsing the LLM’s responses required engineering strict output formats, validating them, and handling errors when the model deviated from the expected structure. With function calling, the model now natively supports structured responses, ensuring we receive either a valid function call or a standard text reply.

This dramatically reduces the complexity of parsing responses and handling actions. Instead of manually extracting commands from unstructured text, we can directly interpret the function calls provided by the model and execute them with confidence.

How Function Calling Simplifies the Agent Loop
Let’s revisit the agent loop, but now using function calling. Instead of handling multiple layers of parsing, we let the model’s API return structured function calls when needed.

import json
import os
from typing import List

from litellm import completion

def list_files() -> List[str]:
    """List files in the current directory."""
    return os.listdir(".")

def read_file(file_name: str) -> str:
    """Read a file's contents."""
    try:
        with open(file_name, "r") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: {file_name} not found."
    except Exception as e:
        return f"Error: {str(e)}"

def terminate(message: str) -> None:
    """Terminate the agent loop and provide a summary message."""
    print(f"Termination message: {message}")

tool_functions = {
    "list_files": list_files,
    "read_file": read_file,
    "terminate": terminate
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Returns a list of files in the directory.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the content of a specified file in the directory.",
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
            "description": "Terminates the conversation. No further actions or interactions are possible after this. Prints the provided message for the user.",
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

agent_rules = [{
    "role": "system",
    "content": """
You are an AI agent that can perform tasks by using available tools. 

If a user asks about files, documents, or content, first list the files before reading them.

When you are done, terminate the conversation by using the "terminate" tool and I will provide the results to the user.
"""
}]

# Initialize agent parameters
iterations = 0
max_iterations = 10

user_task = input("What would you like me to do? ")

memory = [{"role": "user", "content": user_task}]

# The Agent Loop
while iterations < max_iterations:

    messages = agent_rules + memory

    response = completion(
        model="openai/gpt-4o",
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
                result = {"error":f"Error executing {tool_name}: {str(e)}"}
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        print(f"Executing: {tool_name} with args {tool_args}")
        print(f"Result: {result}")
        memory.extend([
            {"role": "assistant", "content": json.dumps(action)},
            {"role": "user", "content": json.dumps(result)}
        ])
    else:
        result = response.choices[0].message.content
        print(f"Response: {result}")
        break

What This Changes

No More Custom Parsing Logic - We don’t need to engineer strict text output parsing; the model always returns structured JSON for function calls.

Dynamic Execution - Instead of a rigid loop that manually checks what the agent should do, we simply read and execute the function call.

Unified Text & Action Handling - If no function call is needed, the model responds with a message, allowing mixed conversational and action-driven workflows.

Automated Function Execution - The agent dynamically maps the tool name from the model to its corresponding Python function and executes it with the provided arguments.


Errors Can Still Occur

While function calling improves structured execution, it does not eliminate all potential issues. One common problem is that the model may still sometimes return improperly formatted JSON or a tool call that isn’t valid. For example, the response could contain syntax errors, missing required fields, or incorrectly formatted values. If the agent attempts to parse this malformed JSON, it will fail.

To address this, we could add error handling around json.loads() to catch json.JSONDecodeError. If a parsing error occurs, the agent retries by sending another request. This prevents the entire loop from crashing and ensures robustness in execution. You can do this as a practice exercise to enhance the agent’s reliability.

Additionally, even when the JSON is correctly formatted, errors can still arise at runtime due to missing files, incorrect arguments, or unexpected edge cases in function execution. Our error handling ensures that these issues do not halt execution entirely but instead provide meaningful feedback for debugging and improvement.

Key Takeaways

By leveraging function calling, we remove unnecessary complexity from the agent loop, allowing the AI to interact with its environment more reliably. This simplification makes AI agents more robust, scalable, and easier to integrate into real-world applications.

In a later lesson, we will explore how to register tools dynamically with decorators, further improving flexibility and maintainability in AI agent design.


Exercise: Extend the Function Calling Agent

This is an optional exercise to practice.

The agent can't do a lot right now in a notebook. See if you can extend the agent to be able to list the contents of a directory. Then, try putting a small number of files with textual information in a directory and asking the agent to "read all the files in <fill in the directory path> and tell me what they are". 

You can copy this notebook to get started:

https://colab.research.google.com/drive/16TU03Dgubcuo6OPCL0mOWuLJ8cxpgL2V?usp=sharing



Module 3


Designing AI Agents with GAME

Motivating Example: The Proactive Coder
To illustrate how the GAME framework applies in practice, consider an AI agent designed to proactively enhance a codebase. This Proactive Coder agent will scan a repository, analyze patterns in the code, and propose potential new features that it could implement with a small number of changes. If the user approves a feature, the agent will generate the initial implementation and suggest refinements.

Using the GAME framework, we break down the agent design:

Goals:

Goals (What to achieve):

Identify potential enhancements
Make sure that the enhancements are helpful and relevant
Make sure that the enhancements are small and self-contained so that they can be implemented by the agent with minimal risk
Ensure that the changes do not break existing interfaces
Ensure that the agent only implements features that the user agrees to

Instructions (How to achieve it):

Pick a random file in the code base and read through it
Read some related files to the original file
Read at most 5 files
Propose three feature ideas that are implementable in 2-3 functions and require minimal editing of the existing code
Ask the user to select which feature to implement
List the files that will need to be edited and provide a list of proposed changes for each
Go file by file implementing the changes until they are all edited

Actions:

List project files
Read project file
Ask user to select a feature
Edit project file

Memory:

We will use a simple conversational memory and store the complete contents of files in the conversation for reference
Environment:

We will provide simple implementations of the actions in Python to run locally, but could later change to an implementation that works in GitHub Actions.

AI Agents and Agentic AI with Python & Generative AI

Module 3

Simulating GAME Agents in Conversation

Testing Agent Designs Through Conversation Simulation
Before we write a single line of code for our agent, we should test whether our GAME design is actually feasible. One powerful technique is to simulate the agent’s decision-making process through conversation with an LLM in a chat interface (e.g., ChatGPT). This approach helps us identify potential problems with our design early, when they’re easiest to fix. Let’s explore how to conduct these simulations effectively.

Why Simulate First?
Think of agent simulation like a dress rehearsal for a play. Before investing in costumes and sets, you want to make sure the script makes sense and the actors can perform their roles effectively. Similarly, before implementing an agent, we want to verify that:

The goals are achievable with the planned actions
The memory requirements are reasonable
The actions available are sufficient to solve the problem
The agent can make appropriate decisions with the available information
Setting Up Your Simulation
When starting a conversation with an LLM to simulate your agent, begin by establishing the framework. We can do this with a simple prompt in a chat interface. The prompt should clearly outline the agent’s goals, actions, and the simulation process. Here’s a template you can use:

I'd like to simulate an AI agent that I'm designing. The agent will be built using these components:

Goals: [List your goals]
Actions: [List available actions]

At each step, your output must be an action to take.

Stop and wait and I will type in the result of
the action as my next message.

Ask me for the first task to perform.
For a Proactive Coder agent, you might use the following prompt to kick-off a simulation in ChatGPT:

I'd like to simulate an AI agent that I'm designing. The agent will be built using these components:

Goals:
* Find potential code enhancements
* Ensure changes are small and self-contained
* Get user approval before making changes
* Maintain existing interfaces

Actions available:
* list_project_files()
* read_project_file(filename)
* ask_user_approval(proposal)
* edit_project_file(filename, changes)

At each step, your output must be an action to take. 

Stop and wait and I will type in the result of 
the action as my next message.

Ask me for the first task to perform.
Take a moment to open up ChatGPT and try out this prompt. You can use the same prompt in any chat interface that supports LLMs. What worked? What didn’t?

Learning Through Agent Simulation
Understanding Agent Reasoning
When you begin simulating your agent’s behavior, you’re essentially conducting a series of experiments to understand how well it can reason with the tools and goals you’ve provided. Start by presenting a simple scenario – perhaps a small Python project with just a few files. Watch how the agent approaches the task. Does it immediately jump to reading files, or does it first list the available files to get an overview? These initial decisions reveal a lot about whether your goals and actions enable systematic problem-solving.

As you observe the agent’s decisions, you’ll notice that the way you present information significantly impacts its reasoning. For instance, when you return the results of list_project_files(), you might first try returning just the filenames:

["main.py", "utils.py", "data_processor.py"]
Then experiment with providing more context:

{
    "files": ["main.py", "utils.py", "data_processor.py"],
    "total_files": 3,
    "directory": "/project"
}
You might discover that the additional metadata helps the agent make more informed decisions about which files to examine next. This kind of experimentation with result formats helps you understand how much context your agent needs to reason effectively.

Evolving Your Tools and Goals

The simulation process often reveals that your initial tool descriptions aren’t as clear as you thought. For example, you might start with a simple description for read_project_file():

read_project_file(filename) -> Returns the content of the specified file
Through simulation, you might find the agent using it incorrectly, leading you to enhance the description:

read_project_file(filename) -> Returns the content of a Python file from the project directory. 
The filename should be one previously returned by list_project_files().
Similarly, your goals might evolve. You might start with “Find potential code enhancements” but discover through simulation that the agent needs more specific guidance. This might lead you to refine the goal to “Identify opportunities to improve error handling and input validation in functions.”

Understanding Memory Through Chat
One of the most enlightening aspects of simulation is realizing that the chat format naturally mimics the list-based memory system we use in our agent loop memory. Each exchange between you and the LLM represents an iteration of the agent loop and a new memory entry – the agent’s actions and the environment’s responses accumulate just as they would in our implemented memory system. This helps you understand how much history the agent can accumulate and still maintain context and make good decisions.

Learning from Failures
Introducing controlled chaos into your simulation provides valuable insights. Try returning error messages instead of successful results:

{"error": "FileNotFoundError: main.py does not exist"}
Or return malformed data:

{"cont3nt": "def broken_func(): pass"}
Watch how the agent handles these situations. Does it try alternative approaches? Does it give up too easily? Does it maintain its goal focus despite errors? These observations help you design better error handling and recovery strategies.

Preventing Runaway Agents
The simulation environment provides a safe space to test termination conditions. You can experiment with different criteria for when the agent should conclude its task. Perhaps it should stop after examining a certain number of files, or after making a specific number of improvement suggestions. The chat format lets you quickly try different approaches without worrying about infinite loops or resource consumption.

Rapid Iteration and Improvement
The true power of simulation lies in its speed. You can test dozens of scenarios in the time it would take to implement a single feature. Want to see how the agent handles a project with 100 files? Just tell it that’s what list_project_files() returned. Curious about how it would handle deeply nested function calls? Paste in some complex code and see how it analyzes it.

Learning from the Agent

At the end of your simulation sessions, ask the agent to reflect on its experience. What tools did it wish it had? Were any instructions unclear? Which goals were too vague? The LLM can often provide surprisingly insightful suggestions about how to improve your GAME design.

For example, the agent might suggest: “The ask_user_approval() action would be more effective if it could include code snippets showing the proposed changes. This would help users make more informed decisions about the suggested improvements.”

Building Your Example Library
As you conduct these simulations, you’re building a valuable library of examples. When you see the agent make a particularly good decision, save that exchange. When it makes a poor choice, save that too. These examples become invaluable when you move to implementation – they can be used to craft better prompts and test cases.

Keep a record of exchanges like this:

Good Example:

Agent: "Before modifying utils.py, I should read its contents to understand the current error handling patterns."
Action: read_project_file("utils.py")
Result: [file contents]
Agent: "I notice these functions lack input validation. I'll propose focused improvements for each function."
Poor Example:

Agent: "I'll start editing all the files to add error handling."
Action: edit_project_file("utils.py", {...})
[Missing analysis and user approval steps]
These examples help you understand what patterns to encourage or discourage in your implemented agent.

Through this iterative process of simulation, observation, and refinement, you develop a deep understanding of how your agent will behave in the real world. This understanding is invaluable when you move to implementation, helping you build agents that are more robust, more capable, and better aligned with your goals.

Remember, the time spent in simulation is an investment that pays off in better design decisions and fewer implementation surprises. When you finally start coding, you’re not just hoping your design will work – you’ve already seen it work in hundreds of scenarios.

AI Agents and Agentic AI with Python & Generative AI
Module 3
Modular AI Agent Design

Building a Simple Agent Framework 1
We are designing our agents in terms of GAME. Ideally, we would like our code to reflect how we design the agent, so that we can easily translate our design into an implementation. Also, we can see that the GAME components are what change from one agent to another while the core loop stays the same. We would like to design a framework that allows us to reuse as much as possible while making it easy to change the GAME pieces without affecting the GAME rules (e.g., the agent loop).

At first, it will appear that we are adding complexity to the agent — and we are. However, this complexity is necessary to create a framework that is flexible and reusable. The goal is to create a framework that allows us to build agents quickly and easily without changing the core loop. We are going to look at each of the individual GAME component implementations and then how they fit into the overall framework at the end.

G - Goals Implementation
First, let’s create a simple goal class that defines what our agent is trying to accomplish:

@dataclass(frozen=True)
class Goal:
    priority: int
    name: str
    description: str

| Concept            | Explanation                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| `@dataclass`       | Automates creation of boilerplate methods (`__init__`, `__repr__`, etc.) |
| `frozen=True`      | Makes the object immutable after creation                                |
| `priority: int`    | Integer indicating the goal's priority                                   |
| `name: str`        | Short name or title of the goal                                          |
| `description: str` | Longer explanation of the goal                                           |


| Concept          | Description                                                        |
| ---------------- | ------------------------------------------------------------------ |
| `__repr__`       | Special method to return a string that **represents the object**   |
| Purpose          | For developers; should be unambiguous and ideally executable       |
| `@dataclass` use | Auto-generates `__repr__` based on fields if `repr=True` (default) |

🔹 What is __repr__ in Python?
__repr__ stands for representation. It's a special method (a “dunder” method — double underscores) used to define how an object is represented as a string, primarily for debugging and development purposes.

Syntax:

python

def __repr__(self) -> str:
    return "<some string representation>"



Goals will describe what we are trying to achieve and how to achieve it. By encapsulating them into objects, we can move away from large “walls of text” that represent the instructions for our agent. Additionally, we can add priority to our goals, which will help us decide which goal to pursue first and how to sort or format them when combining them into a prompt.

We broadly use the term “goal” to encompass both “what” the agent is trying to achieve and “how” it should approach the task. This duality is crucial for guiding the agent’s behavior effectively. An important type of goal can be examples that show the agent how to reason in certain situations. We can also build goals that define core rules that are common across all agents in our system or that give it special instructions on how to solve certain types of tasks.

Now, let’s take a look at how we might create a goal related to file management for our agent:

from game.core import Goal

# Define a simple file management goal
file_management_goal = Goal(
    priority=1,
    name="file_management",
    description="""Manage files in the current directory by:
    1. Listing files when needed
    2. Reading file contents when needed
    3. Searching within files when information is required
    4. Providing helpful explanations about file contents"""
)
A - Actions Implementation with JSON Schemas
Actions define what the agent can do. Think of them as the agent’s toolkit. Each action is a discrete capability that can be executed in the environment. The action system has two main parts: the Action class and the ActionRegistry.

The actions are the interface between our agent and its environment. These are descriptions of what the agent can do to affect the environment. We have previously built out actions using Python functions, but let’s encapsulate the parts of an action into an object:

class Action:
    def __init__(self,
                 name: str,
                 function: Callable,
                 description: str,
                 parameters: Dict,
                 terminal: bool = False):
        self.name = name
        self.function = function
        self.description = description
        self.terminal = terminal
        self.parameters = parameters

    def execute(self, **args) -> Any:
        """Execute the action's function"""
        return self.function(**args)

| Component     | Purpose                                                              |
| ------------- | -------------------------------------------------------------------- |
| `name`        | Label for the action                                                 |
| `function`    | Callable that will be executed                                       |
| `description` | Explains what the action does                                        |
| `parameters`  | Dictionary of expected parameters (could be used for UI, validation) |
| `terminal`    | Indicates if the action is final in a flow                           |
| `execute()`   | Dynamically invokes the function using named arguments (`**args`)    |



At first, it may not appear that this is much different from the previous implementation. However, later, we will see that this makes it much easier to create different agents by simply swapping out the actions without having to modify the core loop.

When the agent provides a response, it is going to return JSON. However, we are going to want a way to lookup the actual object associated with the action indicated by the JSON. To do this, we will create an ActionRegistry that will allow us to register actions and look them up by name:

class ActionRegistry:
    def __init__(self):
        self.actions = {}

    def register(self, action: Action):
        self.actions[action.name] = action

    def get_action(self, name: str) -> [Action, None]:
        return self.actions.get(name, None)

    def get_actions(self) -> List[Action]:
        """Get all registered actions"""
        return list(self.actions.values())

| Method / Attribute | Purpose                                                       |
| ------------------ | ------------------------------------------------------------- |
| `self.actions`     | A dictionary to store actions using their name as the key     |
| `register(action)` | Adds a new action to the registry                             |
| `get_action(name)` | Returns the action with the given name or `None` if not found |
| `get_actions()`    | Returns a list of all registered action objects               |




Here is an example of how we might define some actions for a file management agent:

def list_files() -> list:
    """List all files in the current directory."""
    return os.listdir('.')

def read_file(file_name: str) -> str:
    """Read and return the contents of a file."""
    with open(file_name, 'r') as f:
        return f.read()

def search_in_file(file_name: str, search_term: str) -> list:
    """Search for a term in a file and return matching lines."""
    results = []
    with open(file_name, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if search_term in line:
                results.append((i+1, line.strip()))
    return results

# Create and populate the action registry
registry = ActionRegistry()

registry.register(Action(
    name="list_files",
    function=list_files,
    description="List all files in the current directory",
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    },
    terminal=False
))

registry.register(Action(
    name="read_file",
    function=read_file,
    description="Read the contents of a specific file",
    parameters={
        "type": "object",
        "properties": {
            "file_name": {
                "type": "string",
                "description": "Name of the file to read"
            }
        },
        "required": ["file_name"]
    },
    terminal=False
))

registry.register(Action(
    name="search_in_file",
    function=search_in_file,
    description="Search for a term in a specific file",
    parameters={
        "type": "object",
        "properties": {
            "file_name": {
                "type": "string", 
                "description": "Name of the file to search in"
            },
            "search_term": {
                "type": "string",
                "description": "Term to search for"
            }
        },
        "required": ["file_name", "search_term"]
    },
    terminal=False
))
M - Memory Implementation
Almost every agent needs to remember what happens from one loop iteration to the next. This is where the Memory component comes in. It allows the agent to store and retrieve information about its interactions, which is critical for context and decision-making. We can create a simple class to represent the memory:

class Memory:
    def __init__(self):
        self.items = []  # Basic conversation history

    def add_memory(self, memory: dict):
        """Add memory to working memory"""
        self.items.append(memory)

    def get_memories(self, limit: int = None) -> List[Dict]:
        """Get formatted conversation history for prompt"""
        return self.items[:limit]

| Component             | Type         | Purpose                                                 |
| --------------------- | ------------ | ------------------------------------------------------- |
| `self.items`          | `List[dict]` | Stores the conversation or memory records               |
| `add_memory(memory)`  | method       | Adds a new memory (a dictionary) to the memory list     |
| `get_memories(limit)` | method       | Returns a list of memory items, up to the given `limit` |



Originally, we just used a simple list of messages. Is it worth wrapping the list in this additional class? Yes, because it allows us to add additional functionality later without changing the core loop. For example, we might want to store the memory in a database and dynamically change what memories the agent sees at each loop iteration based on some analysis of the state of the memory. With this simple interface, we can create subclasses that implement different memory strategies without changing the core loop.

One thing to note is that our memory always has to be represented as a list of messages in the prompt. Because of this, we provide a simple interface to the memory that returns the last N messages in the correct format. This allows us to keep the memory class agnostic to how it is used. We can change how we store the memory (e.g., in a database) without changing how we access it in the agent loop. Even if we store the memory in a complicated graph structure, we are still going to need to pass the memories to the LLM as a list and format them as messages.

E - Environment Implementation
In our original implementation, we hardcoded our “environment” interface as a series of if/else statements and function calls. We would like to have a more modular interface that allows us to execute actions without needing to know how they are implemented or have conditional logic in the loop. This is where the Environment component comes in. It serves as a bridge between the agent and the outside world, executing actions and returning results.

class Environment:
    def execute_action(self, action: Action, args: dict) -> dict:
        """Execute an action and return the result."""
        try:
            result = action.execute(**args)
            return self.format_result(result)
        except Exception as e:
            return {
                "tool_executed": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def format_result(self, result: Any) -> dict:
        """Format the result with metadata."""
        return {
            "tool_executed": True, # tool_executed: False (indicates failure)
            "result": result,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z") # traceback: Full error stack trace, useful for debugging
        }

| Key             | Purpose                                                 |
| --------------- | ------------------------------------------------------- |
| `tool_executed` | Boolean flag to indicate success (`True`)               |
| `result`        | The actual return value of the function                 |
| `timestamp`     | Current time in ISO 8601-like format (UTC offset aware) |


    | Method             | Purpose                                                         |
| ------------------ | --------------------------------------------------------------- |
| `execute_action()` | Executes the given action safely, handling errors if they occur |
| `format_result()`  | Wraps result with metadata like execution flag and timestamp    |


AI Agents and Agentic AI with Python & Generative AI

Module 3

Agent Loop Customization

Building a Simple Agent Framework 2
Now, we are going to put the components together into a reusable agent class. This class will encapsulate the GAME components and provide a simple interface for running the agent loop. The agent will be responsible for constructing prompts, executing actions, and managing memory. We can create different agents simply by changing the goals, actions, and environment without modifying the core loop.

Let’s take a look at our agent class:

class Agent:
    def __init__(self,
                 goals: List[Goal],
                 agent_language: AgentLanguage,
                 action_registry: ActionRegistry,
                 generate_response: Callable[[Prompt], str],
                 environment: Environment):
        """
        Initialize an agent with its core GAME components
        """
        self.goals = goals
        self.generate_response = generate_response
        self.agent_language = agent_language
        self.actions = action_registry
        self.environment = environment

    def construct_prompt(self, goals: List[Goal], memory: Memory, actions: ActionRegistry) -> Prompt:
        """Build prompt with memory context"""
        return self.agent_language.construct_prompt(
            actions=actions.get_actions(),
            environment=self.environment,
            goals=goals,
            memory=memory
        )

    def get_action(self, response):
        invocation = self.agent_language.parse_response(response)
        action = self.actions.get_action(invocation["tool"])
        return action, invocation

    def should_terminate(self, response: str) -> bool:
        action_def, _ = self.get_action(response)
        return action_def.terminal

    def set_current_task(self, memory: Memory, task: str):
        memory.add_memory({"type": "user", "content": task})

    def update_memory(self, memory: Memory, response: str, result: dict):
        """
        Update memory with the agent's decision and the environment's response.
        """
        new_memories = [
            {"type": "assistant", "content": response},
            {"type": "user", "content": json.dumps(result)}
        ]
        for m in new_memories:
            memory.add_memory(m)

    def prompt_llm_for_action(self, full_prompt: Prompt) -> str:
        response = self.generate_response(full_prompt)
        return response

    def run(self, user_input: str, memory=None, max_iterations: int = 50) -> Memory:
        """
        Execute the GAME loop for this agent with a maximum iteration limit.
        """
        memory = memory or Memory()
        self.set_current_task(memory, user_input)

        for _ in range(max_iterations):
            # Construct a prompt that includes the Goals, Actions, and the current Memory
            prompt = self.construct_prompt(self.goals, memory, self.actions)

            print("Agent thinking...")
            # Generate a response from the agent
            response = self.prompt_llm_for_action(prompt)
            print(f"Agent Decision: {response}")

            # Determine which action the agent wants to execute
            action, invocation = self.get_action(response)

            # Execute the action in the environment
            result = self.environment.execute_action(action, invocation["args"])
            print(f"Action Result: {result}")

            # Update the agent's memory with information about what happened
            self.update_memory(memory, response, result)

            # Check if the agent has decided to terminate
            if self.should_terminate(response):
                break

        return memory
Now, let’s walk through how the GAME components work together in this agent architecture, explaining each part of agent loop.

Step 1: Constructing the Prompt
When the agent loop begins, it first constructs a prompt using the construct_prompt method:

def construct_prompt(self, goals: List[Goal], memory: Memory, actions: ActionRegistry) -> Prompt:
    """Build prompt with memory context"""
    return self.agent_language.construct_prompt(
        actions=actions.get_actions(),
        environment=self.environment,
        goals=goals,
        memory=memory
    )
This method leverages the AgentLanguage component to build a structured prompt containing:

The agent’s goals (what it’s trying to accomplish)
Available actions (tools the agent can use)
Current memory context (conversation history and relevant information)
Environment details (constraints and context for operation)

We are going to discuss the AgentLanguage in more detail later. For now, what you need to know is that it is responsible for formatting the prompt that is sent to the LLM and parsing the response from the LLM. Most of the time, we are going to use function calling, so the parsing will just be reading the returned tool calls. However, the AgentLanguage can be changed to allow us to also take the same agent and implement it without function calling.

Step 2: Generating a Response
Next, the agent sends this prompt to the language model:

def prompt_llm_for_action(self, full_prompt: Prompt) -> str:
    response = self.generate_response(full_prompt)
    return response
The generate_response function is a simple python function provided during initialization. This abstraction allows the framework to work with different language models without changing the core loop. We will use LiteLLM to call the LLM, but you could easily swap this out for any other LLM provider.

Step 3: Parsing the Response
Once the language model returns a response, the agent parses it to identify the intended action. The parsing will generally be just getting the tool calls from the response, however the agent language gets to decide how this is done. Once the response is parsed, the agent can look up the action in the ActionRegistry:

def get_action(self, response):
    invocation = self.agent_language.parse_response(response)
    action = self.actions.get_action(invocation["tool"])
    return action, invocation
The action is the interface definition of what the agent “can” do. The invocation is the specific parameters that the agent has chosen to use for this action. The ActionRegistry allows the agent to look up the action by name, and the invocation provides the arguments needed to execute it. We could also add validation at this step to ensure that the invocation parameters match the action’s expected parameters.

Step 4: Executing the Action
The agent then executes the chosen action in the environment:

# Execute the action in the environment
result = self.environment.execute_action(action, invocation["args"])
The Environment handles the actual execution of the action, which might involve:

Making API calls
Reading/writing files
Querying databases
Processing data
Actions are defined in the ActionRegistry but executed within the context of the Environment, which provides access to resources and handles the mechanics of execution.

Step 5: Updating Memory
After execution, the agent updates its memory with both its decision and the result:

def update_memory(self, memory: Memory, response: str, result: dict):
    """
    Update memory with the agent's decision and the environment's response.
    """
    new_memories = [
        {"type": "assistant", "content": response},
        {"type": "user", "content": json.dumps(result)}
    ]
    for m in new_memories:
        memory.add_memory(m)
This creates a continuous record of the agent’s reasoning and actions, which becomes part of the context of future loop iterations. The memory serves both as a record of past actions and as context for future prompt construction.

Step 6: Termination Check
Finally, the agent checks if it should terminate the loop:

def should_terminate(self, response: str) -> bool:
    action_def, _ = self.get_action(response)
    return action_def.terminal
This allows certain actions (like a “terminate” action) to signal that the agent has finished its work.

The Flow of Information Through the Loop
To better understand how these components interact, let’s trace how information flows through a single iteration of the loop:

The Memory provides context about what the user has asked the agent to do and past decisions and results from the agent loop
The Goals define what the agent is trying to accomplish and rules on how to accomplish it
The ActionRegistry defines what the agent can do and helps lookup the action to execute by name
The AgentLanguage formats Memory, Actions, and Goals into a prompt for the LLM
The LLM generates a response choosing an action
The AgentLanguage parses the response into an action invocation, which will typically be extracted from tool calls
The Environment executes the action with the given arguments
The result is stored back in Memory
The loop repeats with the updated memory until the agent calls a terminal tool or reaches the maximum number of iterations
Creating Specialized Agents
The beauty of this framework is that we can create entirely different agents by changing the GAME components without modifying the core loop:

# A research agent
research_agent = Agent(
    goals=[Goal("Find and summarize information on topic X")],
    agent_language=ResearchLanguage(),
    action_registry=ActionRegistry([SearchAction(), SummarizeAction(), ...]),
    generate_response=openai_call,
    environment=WebEnvironment()
)

# A coding agent
coding_agent = Agent(
    goals=[Goal("Write and debug Python code for task Y")],
    agent_language=CodingLanguage(),
    action_registry=ActionRegistry([WriteCodeAction(), TestCodeAction(), ...]),
    generate_response=anthropic_call,
    environment=DevEnvironment()
)
Each agent operates using the same fundamental loop but exhibits completely different behaviors based on its GAME components.

AI Agents and Agentic AI with Python & Generative AI

Module 3

Implementing GAME in Code

Building a Simple Agent Framework 3
Let’s go back to the file agent that we built earlier. The original implementation uses direct function calls and a lot of conditional logic in the agent loop. Let’s redo the implementation using our new framework.

Define the Goals
First, let’s define some goals for our file explorer agent:

# Define clear goals for the agent
goals = [
    Goal(
        priority=1, 
        name="Explore Files", 
        description="Explore files in the current directory by listing and reading them"
    ),
    Goal(
        priority=2, 
        name="Terminate", 
        description="Terminate the session when tasks are complete with a helpful summary"
    )
]
Create Actions Using the Framework
Next, let’s convert our tool functions into properly structured Actions in our AgentRegistry:

def list_files() -> List[str]:
    """List files in the current directory."""
    return os.listdir(".")

def read_file(file_name: str) -> str:
    """Read a file's contents."""
    try:
        with open(file_name, "r") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: {file_name} not found."
    except Exception as e:
        return f"Error: {str(e)}"

def terminate(message: str) -> str:
    """Terminate the agent loop and provide a summary message."""
    return message

# Create and register the actions
action_registry = ActionRegistry()

action_registry.register(Action(
    name="list_files",
    function=list_files,
    description="Returns a list of files in the directory.",
    parameters={},
    terminal=False
))

action_registry.register(Action(
    name="read_file",
    function=read_file,
    description="Reads the content of a specified file in the directory.",
    parameters={
        "type": "object",
        "properties": {
            "file_name": {"type": "string"}
        },
        "required": ["file_name"]
    },
    terminal=False
))

action_registry.register(Action(
    name="terminate",
    function=terminate,
    description="Terminates the conversation. Prints the provided message for the user.",
    parameters={
        "type": "object",
        "properties": {
            "message": {"type": "string"},
        },
        "required": ["message"]
    },
    terminal=True
))

Create and Run the Agent

Now we can put it all together:


# Create the agent
file_explorer_agent = Agent(
    goals=goals,
    agent_language=agent_language,
    action_registry=action_registry,
    generate_response=generate_response,
    environment=environment
)

# Run the agent
user_input = input("What would you like me to do? ")
final_memory = file_explorer_agent.run(user_input, max_iterations=10)

# Print the final conversation if desired
for item in final_memory.get_memories():
    print(f"\n{item['type'].upper()}: {item['content']}")
Complete Implementation
Here’s the full implementation using the GAME framework:

def main():
    # Define the agent's goals
    goals = [
        Goal(
            priority=1, 
            name="Explore Files", 
            description="Explore files in the current directory by listing and reading them"
        ),
        Goal(
            priority=2, 
            name="Terminate", 
            description="Terminate the session when tasks are complete with a helpful summary"
        )
    ]
    
    # Define tool functions
    def list_files() -> List[str]:
        """List files in the current directory."""
        return os.listdir(".")

    def read_file(file_name: str) -> str:
        """Read a file's contents."""
        try:
            with open(file_name, "r") as file:
                return file.read()
        except FileNotFoundError:
            return f"Error: {file_name} not found."
        except Exception as e:
            return f"Error: {str(e)}"

    def terminate(message: str) -> str:
        """Terminate the agent loop and provide a summary message."""
        return message
    
    # Create action registry and register actions
    action_registry = ActionRegistry()
    
    action_registry.register(Action(
        name="list_files",
        function=list_files,
        description="Returns a list of files in the directory.",
        parameters={},
        terminal=False
    ))
    
    action_registry.register(Action(
        name="read_file",
        function=read_file,
        description="Reads the content of a specified file in the directory.",
        parameters={
            "type": "object",
            "properties": {
                "file_name": {"type": "string"}
            },
            "required": ["file_name"]
        },
        terminal=False
    ))
    
    action_registry.register(Action(
        name="terminate",
        function=terminate,
        description="Terminates the conversation. Prints the provided message for the user.",
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"]
        },
        terminal=True
    ))
    
    # Define the agent language and environment
    agent_language = AgentFunctionCallingActionLanguage()
    environment = Environment()
    
    # Create the agent
    file_explorer_agent = Agent(
        goals=goals,
        agent_language=agent_language,
        action_registry=action_registry,
        generate_response=generate_response,
        environment=environment
    )
    
    # Run the agent
    user_input = input("What would you like me to do? ")
    final_memory = file_explorer_agent.run(user_input, max_iterations=10)
    
    # Print the termination message (if any)
    for item in final_memory.get_memories():
        print(f"\nMemory: {item['content']}")

if __name__ == "__main__":
    main()

Key Differences and Benefits
By converting our agent to the GAME framework, we gain several benefits:

Better Organization: Each component has a clear purpose and is separated from others.
Reusability: We can swap out components (like the actions or environment) without changing the core logic.
Extensibility: New goals and actions can be added easily.
Standard Interface: Using the Agent class gives us a consistent way to interact with different agents.
Memory Management: The framework handles memory updates automatically.
This structure also makes it easier to understand and maintain the code, especially as your agent grows in complexity.

Using the Agent
Once implemented, you can use your file explorer agent like this:

What would you like me to do? Tell me what Python files are in this directory and summarize how they fit together.

Agent thinking...
Agent Decision: I'll help you explore the Python files in this directory.

{"tool_name": "list_files", "args": {}}

Action Result: {'tool_executed': True, 'result': ['file1.py', 'file2.py', 'main.py', ...], 'timestamp': '2025-03-02T12:34:56+0000'}

{"tool_name": "read_file", "args": {"file_name": "file1.py"}}

Action Result: {'tool_executed': True, 'result': '# This is file1.py\n\ndef hello_world():\n    print("Hello, World!")\n\nif __name__ == "__main__":\n    hello_world()', 'timestamp': '2025-03-02T12:34:58+0000'}

[Additional file readings...]

{"tool_name": "terminate", "args": {"message": "I've explored all Python files in this directory. Here's a summary: file1.py contains a simple hello_world function, file2.py implements a calculator class, and main.py imports both files and uses their functionality."}}

...
This structured approach makes it much easier to develop, maintain, and extend your agents over time

AI Agents and Agentic AI with Python & Generative AI

Module 3

How Your Agent Communicates with the LLM: The Agent Language

Building a Simple Agent Framework 4
We’ve discussed Goals, Actions, Memory, and Environment, but there’s another crucial component we need to explore: the AgentLanguage. This component serves as the translator between our structured agent components and the language model’s input/output format. Think of it as a diplomatic interpreter that ensures clear communication between two different worlds: our agent’s structured GAME components and the LLM’s text-based interface.

As we have already seen, there are multiple ways that we can prompt the LLM for a next action. For example, we can have the LLM generate a standard completion with text that we parse or use function calling to extract an action. There are also many different ways that we could represent memories to the LLM, from concatenating them into a single string to including them as individual message entries in ChatML. The AgentLanguage allows us to create reusable strategies for handling these concerns and plug them into the agent.

For example, we might define an AgentLanguage that always constructs a system message explaining the agent’s role, followed by a user message containing the agent’s current observations, memory, and a request for the next action. Alternatively, we could use function calling to directly extract structured actions, bypassing the need for parsing. Each of these choices influences how the LLM reasons and responds, shaping the agent’s behavior.

The Role of AgentLanguage
The AgentLanguage component has two primary responsibilities:

Prompt Construction: Transforming our GAME components into a format the LLM can understand
Response Parsing: Interpreting the LLM’s response to determine what action the agent should take
Let’s look at how this works in practice, starting with the base abstract class:

class AgentLanguage:
    def construct_prompt(self,
                        actions: List[Action],
                        environment: Environment,
                        goals: List[Goal],
                        memory: Memory) -> Prompt:
        raise NotImplementedError("Subclasses must implement this method")

    def parse_response(self, response: str) -> dict:
        raise NotImplementedError("Subclasses must implement this method")
This abstract class defines the interface that all agent languages must implement. Let’s examine three different implementations to understand how we can adapt our agent’s communication style.

Where this Fits in the Agent Loop
Let’s examine how the AgentLanguage component integrates with each stage of the loop, transforming raw data into meaningful communication and back again.

Consider this portion of our agent’s run method:

def run(self, user_input: str, memory=None, max_iterations: int = 50) -> Memory:
    memory = memory or Memory()
    self.set_current_task(memory, user_input)

    for _ in range(max_iterations):
        # 1. Build prompt using AgentLanguage
        prompt = self.construct_prompt(self.goals, memory, self.actions)

        # 2. Get LLM response
        response = self.prompt_llm_for_action(prompt)

        # 3. Parse response using AgentLanguage
        action, invocation = self.get_action(response)

        # 4. Execute action in environment
        result = self.environment.execute_action(action, invocation["args"])

        # 5. Update memory
        self.update_memory(memory, response, result)

        if self.should_terminate(response):
            break

    return memory

At two crucial points in this loop, the AgentLanguage acts as an interpreter between our structured world and the LLM’s text-based world:

Stage 1: Constructing the Prompt
When the agent needs to decide its next action, the AgentLanguage takes our GAME components and transforms them into a format the LLM can understand. This transformation involves several steps:

def construct_prompt(self, goals: List[Goal], memory: Memory, actions: ActionRegistry):
    # The AgentLanguage decides how to present each component to the LLM
    prompt = []
    
    # Transform goals into instructions
    prompt += self.format_goals(goals)
    
    # Transform available actions into tool descriptions
    prompt += self.format_actions(actions.get_actions())
    
    # Transform memory into conversation context
    prompt += self.format_memory(memory)
    
    return Prompt(messages=prompt, tools=tools)
For example, when using function calling, this might produce:

{
    "messages": [
        {"role": "system", "content": "Your goal is to process all files..."},
        {"role": "user", "content": "Please analyze file.txt"},
        {"role": "assistant", "content": "I'll read the file..."}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Reads a file from the system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"}
                    }
                }
            }
        }
    ]
}
Stage 2: Parsing the Response
After the LLM generates a response, the AgentLanguage must interpret it to determine what action the agent should take:

def get_action(self, response):
    # AgentLanguage parses the LLM's response into a structured format
    invocation = self.agent_language.parse_response(response)
    
    # The parsed response is used to look up the actual action
    action = self.actions.get_action(invocation["tool"])
    return action, invocation
For instance, when using JSON action format, the AgentLanguage might receive this response from the LLM that mixes the agent’s chatty response with a markdown block containing the specification for the action:

Let me analyze the contents of the file.

```action
{
    "tool": "read_file",
    "args": {
        "file_path": "file.txt"
    }
}
The AgentLanguage would then parse this to extract the JSON and convert it into a structured action:

{
    "tool": "read_file",
    "args": {
        "file_path": "file.txt"
    }
}
The AgentLanguage ensures that regardless of how the LLM prefers to communicate (function calling, JSON blocks, or natural language), the agent’s core loop remains unchanged. It’s like having different translators for different languages – the meaning stays the same, but the way it’s expressed adapts to the audience.

Two Example Agent Languages
Let’s look at two example implementations of the AgentLanguage component, each with a different approach to prompting and parsing. The first is a simple natural language approach, like what we used in our very first agents. The second is a more structured approach that leverages LLM function calling.

JSON Action Language
This language allows the LLM to output text and specify actions in special ```action markdown blocks. This is similar to what we did in our first agent examples:

class AgentJsonActionLanguage(AgentLanguage):
    action_format = """
<Stop and think step by step. Insert your thoughts here.>

```action
{
    "tool": "tool_name",
    "args": {...fill in arguments...}
}
```"""

    def format_actions(self, actions: List[Action]) -> List:
        # Convert actions to a description the LLM can understand
        action_descriptions = [
            {
                "name": action.name,
                "description": action.description,
                "args": action.parameters
            } 
            for action in actions
        ]
        
        return [{
            "role": "system",
            "content": f"""
Available Tools: {json.dumps(action_descriptions, indent=4)}

{self.action_format}
"""
        }]

    def parse_response(self, response: str) -> dict:
        """Extract and parse the action block"""
        try:
            start_marker = "```action"
            end_marker = "```"
            
            stripped_response = response.strip()
            start_index = stripped_response.find(start_marker)
            end_index = stripped_response.rfind(end_marker)
            json_str = stripped_response[
                start_index + len(start_marker):end_index
            ].strip()
            
            return json.loads(json_str)
        except Exception as e:
            print(f"Failed to parse response: {str(e)}")
            raise e
Function Calling Language
This next language uses the LLM’s function calling capabilities to directly specify actions. This approach helps alleviate the burden of parsing free-form text. The downside is that we don’t necessarily get to see the LLM’s reasoning, but the upside is that it simplifies getting valid JSON as output.

class AgentFunctionCallingActionLanguage(AgentLanguage):
    def format_actions(self, actions: List[Action]) -> List:
        """Convert actions to function descriptions"""
        return [
            {
                "type": "function",
                "function": {
                    "name": action.name,
                    "description": action.description[:1024],
                    "parameters": action.parameters,
                },
            } 
            for action in actions
        ]

    def construct_prompt(self,
                        actions: List[Action],
                        environment: Environment,
                        goals: List[Goal],
                        memory: Memory) -> Prompt:
        prompt = []
        prompt += self.format_goals(goals)
        prompt += self.format_memory(memory)
        
        tools = self.format_actions(actions)
        
        return Prompt(messages=prompt, tools=tools)

    def parse_response(self, response: str) -> dict:
        """Parse the function call response"""
        try:
            return json.loads(response)
        except Exception as e:
            return {
                "tool": "terminate",
                "args": {"message": response}
            }

The Power of Swappable Languages
The ability to swap agent languages gives us remarkable flexibility in how our agent communicates. Consider these scenarios:

# Create an agent that uses natural language for simple tasks
simple_agent = Agent(
    goals=goals,
    agent_language=AgentJsonActionLanguage(),
    action_registry=registry,
    generate_response=llm.generate,
    environment=env
)

# Create an agent that uses function calling for complex tasks
complex_agent = Agent(
    goals=goals,
    agent_language=AgentFunctionCallingActionLanguage(),
    action_registry=registry,
    generate_response=llm.generate,
    environment=env
)
The same agent can behave differently just by changing its language implementation. This separation of concerns means we can:

Experiment with different prompt formats without changing the agent’s logic

Support different LLM providers with their own communication styles, allowing us to adjust prompting style to match the LLM’s strengths

Add new response formats without modifying existing code

Handle errors and retry logic at the language level

Wrapping Up

The AgentLanguage component is crucial because it:

Centralizes Communication Logic: All prompt construction and response parsing is in one place

Enables Experimentation: We can try different prompt strategies by creating new language implementations

Improves Reliability: Structured response formats and error handling make the agent more robust

Supports Evolution: As LLM capabilities change, we can adapt our communication approach without changing the agent’s core logic

By separating the “how to communicate” from the “what to do,” we create agents that can evolve and improve their interaction with LLMs while maintaining their core functionality. This flexibility is essential as language model capabilities continue to advance and new communication patterns emerge.

AI Agents and Agentic AI with Python & Generative AI

Module 3

Putting It All Together: Document Your Code with a README Agent

Putting It All Together: Building a Simple README Agent
Now that we understand all the components of our framework, let’s see how they work together by building a simple but practical agent. We’ll create an agent that can analyze Python files in a project and write a README file. This example will demonstrate how our modular design makes it straightforward to assemble an agent from well-defined components.

Understanding Our Agent’s Purpose
Before we dive into the code, let’s understand what we want our agent to do. Our README agent will:

Look for Python files in a project directory
Read the contents of each file
Analyze what it finds
Generate a README based on its analysis
This task is perfect for demonstrating our framework because it requires the agent to make decisions about which files to read, process information iteratively, and produce a final output.

Defining the Goals
Let’s start by defining what our agent should achieve. We use goals to give the agent its purpose and guide its decision-making:

goals = [
    Goal(
        priority=1, 
        name="Gather Information", 
        description="Read each file in the project"
    ),
    Goal(
        priority=1, 
        name="Terminate", 
        description="Call the terminate call when you have read all the files "
                   "and provide the content of the README in the terminate message"
    )
]
Notice how we break the task into two clear goals. The first goal drives the agent to explore the project’s files, while the second ensures it knows when to stop and produce output. We give both goals equal priority (priority=1) since they’re sequential steps in the process.

Creating the Actions
Next, we define what our agent can do by creating its available actions. We need three basic capabilities:

def read_project_file(name: str) -> str:
    with open(name, "r") as f:
        return f.read()

def list_project_files() -> List[str]:
    return sorted([file for file in os.listdir(".")
                  if file.endswith(".py")])

# Register these actions with clear descriptions
action_registry = ActionRegistry()
action_registry.register(Action(
    name="list_project_files",
    function=list_project_files,
    description="Lists all files in the project.",
    parameters={},
    terminal=False
))

action_registry.register(Action(
    name="read_project_file",
    function=read_project_file,
    description="Reads a file from the project.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "required": ["name"]
    },
    terminal=False
))

action_registry.register(Action(
    name="terminate",
    function=lambda message: f"{message}\nTerminating...",
    description="Terminates the session and prints the message to the user.",
    parameters={
        "type": "object",
        "properties": {
            "message": {"type": "string"}
        },
        "required": []
    },
    terminal=True
))

Each action is carefully designed with:

A clear name that describes its purpose
A function that implements the action
A description that helps the LLM understand when to use it
A schema defining its parameters
A terminal flag indicating if it ends the agent’s execution

Choosing the Agent Language
For our README agent, we’ll use the function calling language implementation because it provides the most reliable way to structure the agent’s actions:

agent_language = AgentFunctionCallingActionLanguage()
This choice means our agent will use the LLM’s built-in function calling capabilities to select actions. The AgentLanguage will:

Format our goals as system messages
Present our actions as function definitions
Maintain conversation history in the memory
Parse function calls from the LLM’s responses
Setting Up the Environment
Our environment is simple since we’re just working with local files:

environment = Environment()
We use the default Environment implementation because our actions are straightforward file operations. For more complex agents, we might need to customize the environment to handle specific execution contexts or error cases.

Assembling and Running the Agent
Now we can bring all these components together:

# Create an agent instance with our components
agent = Agent(
    goals=goals,
    agent_language=AgentFunctionCallingActionLanguage(),
    action_registry=action_registry,
    generate_response=generate_response,
    environment=environment
)

# Run the agent with our task
user_input = "Write a README for this project."
final_memory = agent.run(user_input)
When we run this agent, several things happen:

The agent receives the user’s request for a README
It uses list_project_files to discover what files exist
It uses read_project_file to examine each relevant file
When it has gathered enough information, it uses terminate to provide the README content
Understanding the Flow
Let’s walk through a typical execution:

First Iteration:

Agent constructs prompt with goals and available actions
LLM decides to list files first (logical starting point)
Environment executes list_project_files
Memory stores the list of files
Middle Iterations:

Agent includes file list in context
LLM chooses files to read based on their names
Environment executes read_project_file for each chosen file
Memory accumulates file contents
Final Iteration:

Agent determines it has enough information
LLM generates README content
Agent uses terminate action to deliver the result
Making It Work
The modular design means we could easily modify this agent to:

Handle different file types by adding new actions
Generate different documentation by changing the goals
Work with remote files by modifying the environment
Use different LLM providers by changing the agent language
This example demonstrates how our framework’s separation of concerns makes it easy to create focused, task-specific agents. Each component has a clear responsibility, making the code easy to understand and modify. The GAME architecture lets us think about each aspect of the agent’s behavior independently while ensuring they work together seamlessly.

Remember, this is just a starting point. With this foundation, we can build more sophisticated agents by:

Adding more complex actions
Implementing smarter memory management
Creating specialized environments
Developing custom agent languages for specific needs
The key is that our framework makes these extensions possible without having to change the core agent loop or other components. This modularity is what makes our framework both powerful and practical.

The Complete Code for Our Agent


def main():
    # Define the agent's goals
    goals = [
        Goal(priority=1, name="Gather Information", description="Read each file in the project"),
        Goal(priority=1, name="Terminate", description="Call the terminate call when you have read all the files "
                                                       "and provide the content of the README in the terminate message")
    ]

    # Define the agent's language
    agent_language = AgentFunctionCallingActionLanguage()

    def read_project_file(name: str) -> str:
        with open(name, "r") as f:
            return f.read()

    def list_project_files() -> List[str]:
        return sorted([file for file in os.listdir(".") if file.endswith(".py")])

    # Define the action registry and register some actions
    action_registry = ActionRegistry()
    action_registry.register(Action(
        name="list_project_files",
        function=list_project_files,
        description="Lists all files in the project.",
        parameters={},
        terminal=False
    ))
    action_registry.register(Action(
        name="read_project_file",
        function=read_project_file,
        description="Reads a file from the project.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        },
        terminal=False
    ))
    action_registry.register(Action(
        name="terminate",
        function=lambda message: f"{message}\nTerminating...",
        description="Terminates the session and prints the message to the user.",
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
            "required": []
        },
        terminal=True
    ))

    # Define the environment
    environment = Environment()

    # Create an agent instance
    agent = Agent(goals, agent_language, action_registry, generate_response, environment)

    # Run the agent with user input
    user_input = "Write a README for this project."
    final_memory = agent.run(user_input)

    # Print the final memory
    print(final_memory.get_memories())


if __name__ == "__main__":
    main()

AI Agents and Agentic AI with Python & Generative AI

Module 4

Keeping Agent Tools Up to Date with Python Decorators

Tool Decorators: Keeping Code and Documentation in Sync
When building AI agents, one of the most challenging aspects is maintaining accurate documentation about the tools our agent can use. Every time we modify a tool’s parameters or behavior, we need to update its description and parameter schema. If these get out of sync, our agent might try to use tools incorrectly, leading to failures that can be hard to debug. Let’s explore how we can solve this problem elegantly using Python decorators.

The Challenge of Tool Documentation
Imagine we’re writing a tool for our agent to read files. Without any special handling, we might write something like this:

# In our action registry setup
action_registry.register(Action(
    name="read_file",
    function=read_file,
    description="Reads content from a specified file",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string"}
        },
        "required": ["file_path"]
    }
))

# The actual function
def read_file(file_path: str) -> str:
    """Reads and returns the content of a file."""
    with open(file_path, 'r') as f:
        return f.read()
This approach has several problems. If we add a new parameter to read_file, we need to remember to update the parameters schema. If we change the function’s behavior, we need to update the description. It’s easy for these to become inconsistent over time.

The Power of Decorators
Instead, we can use Python’s decorator system to automatically extract and maintain this information. Here’s how it works:

@register_tool(tags=["file_operations"])
def read_file(file_path: str) -> str:
    """Reads and returns the content of a file from the specified path.
    
    The function opens the file in read mode and returns its entire contents
    as a string. If the file doesn't exist or can't be read, it raises an
    appropriate exception.
    
    Args:
        file_path: The path to the file to read
        
    Returns:
        The contents of the file as a string
    """
    with open(file_path, 'r') as f:
        return f.read()
Our decorator examines the function and automatically:

Uses the function name as the tool name
Extracts the docstring for the description
Analyzes type hints and parameters to build the schema
Registers the tool in a central registry
Implementing the Decorator
Let’s look at how this magic happens by first understanding the decorator and then examining the helper function that does the heavy lifting:

def register_tool(tool_name=None, description=None, 
                 parameters_override=None, terminal=False, tags=None):
    """Registers a function as an agent tool."""
    def decorator(func):
        # Extract all metadata from the function
        metadata = get_tool_metadata(
            func=func,
            tool_name=tool_name,
            description=description,
            parameters_override=parameters_override,
            terminal=terminal,
            tags=tags
        )
        
        # Register in our global tools dictionary
        tools[metadata["tool_name"]] = {
            "description": metadata["description"],
            "parameters": metadata["parameters"],
            "function": metadata["function"],
            "terminal": metadata["terminal"],
            "tags": metadata["tags"]
        }
        
        # Also maintain a tag-based index
        for tag in metadata["tags"]:
            if tag not in tools_by_tag:
                tools_by_tag[tag] = []
            tools_by_tag[tag].append(metadata["tool_name"])
        
        return func
    return decorator

This register_tool decorator is designed to transform regular Python functions into callable tools for an agent system. Here’s what’s happening in detail:

The decorator accepts several optional parameters that configure how the function will be registered as a tool.
It returns an inner function (decorator) that takes the actual function being decorated (func) as its parameter.
Within this inner function, it first calls get_tool_metadata() to analyze and extract all relevant information from the function to turn it into a tool description that the agent can use.
It then registers the tool in a global dictionary called tools, using the tool name as a key. The AgentRegistry can be configured to lookup tools here.
For improved discovery, it also maintains a tag-based index in tools_by_tag, allowing tools to be looked up by category. This tagging allows us to define sets of tools that should be used together.
Finally, it returns the original function unchanged, allowing it to still be called normally while also being available as a tool.
This pattern creates a clean separation between the definition of tools and their registration, making the codebase much more maintainable.

Now, let’s examine the helper function that does the detailed analysis:

def get_tool_metadata(func, tool_name=None, description=None, 
                     parameters_override=None, terminal=False, tags=None):
    """Extracts metadata for a function to use in tool registration."""
    
    # Use function name if no tool_name provided
    tool_name = tool_name or func.__name__
    
    # Use docstring if no description provided
    description = description or (func.__doc__.strip() 
                                if func.__doc__ else "No description provided.")
    
    # If no parameter override, analyze the function
    if parameters_override is None:
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        # Build JSON schema for arguments
        args_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Examine each parameter
        for param_name, param in signature.parameters.items():
            # Skip special parameters
            if param_name in ["action_context", "action_agent"]:
                continue

            # Convert Python types to JSON schema types
            param_type = type_hints.get(param_name, str)
            param_schema = {
                "type": get_json_type(param_type)
            }
            
            args_schema["properties"][param_name] = param_schema
            
            # If parameter has no default, it's required
            if param.default == inspect.Parameter.empty:
                args_schema["required"].append(param_name)
    else:
        args_schema = parameters_override
    
    return {
        "tool_name": tool_name,
        "description": description,
        "parameters": args_schema,
        "function": func,
        "terminal": terminal,
        "tags": tags or []
    }
This helper function performs introspection on the decorated function to automatically extract the metadata needed to describe the tool to the agent:

It determines the tool’s name, defaulting to the function’s name if not explicitly provided.
It extracts a description from the function’s docstring if one isn’t explicitly provided.
For parameter handling, it does sophisticated introspection using Python’s inspect and typing modules:
It captures the function’s signature to identify all parameters
It extracts type annotations using get_type_hints()
It builds a JSON schema that describes the expected input structure
It identifies required parameters (those without default values)
It intelligently skips special context parameters like action_context and action_agent
It converts Python types to JSON schema types via a helper function get_json_type()
It packages all this metadata into a dictionary that the decorator can use to register the tool.
This thorough introspection enables tools to be defined with minimal boilerplate while providing rich metadata for the agent system to understand how to call each tool properly. The tool description will always match the function’s signature and docstring, ensuring that the agent has the most accurate information available.

Why Create a Decorator?
You will see this decorator approach in many different agent frameworks. Here is why:

Single Source of Truth: The function itself becomes the authoritative source for all tool information. The docstring describes what it does, the type hints define its parameters, and the implementation shows how it works.

Automatic Updates: When we modify the function’s signature or documentation, the tool registration automatically stays in sync. No more hunting through code to update parameter schemas.

Better Organization: The tags system allows us to categorize tools and find related functionality. We can easily get all “file_operations” tools or all “database_tools”.

Improved Development Experience: We write our tools as normal Python functions with standard documentation. The decorator handles all the complexity of making them available to our agent.

Consider how this simplifies adding a new parameter:

@register_tool(tags=["file_operations"])
def read_file(file_path: str, encoding: str = 'utf-8') -> str:
    """Reads and returns the content of a file.
    
    Args:
        file_path: The path to the file to read
        encoding: The character encoding to use (default: utf-8)
    """
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()
The tool’s parameter schema automatically updates to include the new encoding parameter as optional (since it has a default value). We didn’t need to manually update any registration code or documentation.

Tool Organization for Agents


Organizing Agent Tools: From Tags to Registries
Imagine you’re organizing a workshop. You wouldn’t just throw all your tools into a big box—you’d want to organize them by purpose, perhaps keeping all the measuring tools in one drawer, cutting tools in another, and so on. This is exactly what we’re doing with our agent tools using tags and registries. Let’s explore how this organization system works and how it makes our lives easier when building agents.

Understanding Tool Organization
When we build agents, we often create many tools that serve different purposes. Some tools might handle file operations, others might work with databases, and still others might interact with external APIs. Our organization system has three layers:

Tool Decorators: Tag and document individual tools
Tool Registry: Central storage of all available tools
Action Registry: Curated sets of tools for specific agents
Let’s see how these work together.

Tagging Tools for Organization
First, we use our decorator to tag tools based on their purpose:

@register_tool(tags=["file_operations"])
def read_file(file_path: str) -> str:
    """Reads and returns the content of a file."""
    with open(file_path, 'r') as f:
        return f.read()

@register_tool(tags=["file_operations", "write"])
def write_file(file_path: str, content: str) -> None:
    """Writes content to a file."""
    with open(file_path, 'w') as f:
        f.write(content)

@register_tool(tags=["database", "read"])
def query_database(query: str) -> List[Dict]:
    """Executes a database query and returns results."""
    return db.execute(query)

When we register these tools, our decorator maintains two global registries:

tools: A dictionary of all tools indexed by name
tools_by_tag: A dictionary of tool names organized by tag
# Internal structure of tools_by_tag
{
    "file_operations": ["read_file", "write_file"],
    "write": ["write_file"],
    "database": ["query_database"],
    "read": ["query_database"]
}
This organization allows us to easily find related tools. For instance, we can find all tools related to file operations or all tools that perform read operations.

Creating Focused Action Registries
Now comes the powerful part. When we create an agent, we can easily build an ActionRegistry with just the tools it needs:

def create_file_processing_agent():
    # Create a registry with only file operation tools
    action_registry = ActionRegistry(tags=["file_operations"])
    
    return Agent(
        goals=[Goal(1, "File Processing", "Process project files")],
        agent_language=AgentFunctionCallingActionLanguage(),
        action_registry=action_registry,
        generate_response=generate_response,
        environment=Environment()
    )

def create_database_agent():
    # Create a registry with only database tools
    action_registry = ActionRegistry(tags=["database"])
    
    return Agent(
        goals=[Goal(1, "Database Operations", "Query database as needed")],
        agent_language=AgentFunctionCallingActionLanguage(),
        action_registry=action_registry,
        generate_response=generate_response,
        environment=Environment()
    )

Creating Specialized Agents
We can create agents with very specific tool sets just by specifying tags:

# Create an agent that can only read (no writing)
read_only_agent = Agent(
    goals=[Goal(1, "Read Only", "Read but don't modify data")],
    agent_language=AgentFunctionCallingActionLanguage(),
    action_registry=ActionRegistry(tags=["read"]),
    generate_response=generate_response,
    environment=Environment()
)

# Create an agent that handles all file operations
file_agent = Agent(
    goals=[Goal(1, "File Handler", "Manage file operations")],
    agent_language=AgentFunctionCallingActionLanguage(),
    action_registry=ActionRegistry(tags=["file_operations"]),
    generate_response=generate_response,
    environment=Environment()
)

Refactoring Our README Agent

Refactoring Our README Agent: Using Tool Decorators
In our original README agent, we manually created and registered each action. While this approach works, it requires us to maintain the tool definitions (functions) separately from their metadata (descriptions and parameters). This separation creates opportunities for these two pieces to become out of sync. Let’s improve our agent by using tool decorators to keep everything together and automatically synchronized.

Understanding the Problem
Let’s look at a piece of our original code:

def read_project_file(name: str) -> str:
    with open(name, "r") as f:
        return f.read()

# Later, separately, we define metadata about the function
action_registry.register(Action(
    name="read_project_file",
    function=read_project_file,
    description="Reads a file from the project.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "required": ["name"]
    },
    terminal=False
))
In this code, if we change the function’s parameters, we need to remember to update the parameter schema. If we modify what the function does, we need to remember to update the description. These manual steps are error-prone and can lead to confusion for both developers and the LLM.

The Decorator Solution
Using tool decorators, we can keep all this information together. Here’s how we’ll refactor our README agent:


# First, we'll define our tools using decorators
@register_tool(tags=["file_operations", "read"])
def read_project_file(name: str) -> str:
    """Reads and returns the content of a specified project file.

    Opens the file in read mode and returns its entire contents as a string.
    Raises FileNotFoundError if the file doesn't exist.

    Args:
        name: The name of the file to read

    Returns:
        The contents of the file as a string
    """
    with open(name, "r") as f:
        return f.read()

@register_tool(tags=["file_operations", "list"])
def list_project_files() -> List[str]:
    """Lists all Python files in the current project directory.

    Scans the current directory and returns a sorted list of all files
    that end with '.py'.

    Returns:
        A sorted list of Python filenames
    """
    return sorted([file for file in os.listdir(".")
                   if file.endswith(".py")])

@register_tool(tags=["system"], terminal=True)
def terminate(message: str) -> str:
    """Terminates the agent's execution with a final message.

    Args:
        message: The final message to return before terminating

    Returns:
        The message with a termination note appended
    """
    return f"{message}\nTerminating..."

def main():
    # Define the agent's goals
    goals = [
        Goal(priority=1,
             name="Gather Information",
             description="Read each file in the project in order to build a deep understanding of the project in order to write a README"),
        Goal(priority=1,
             name="Terminate",
             description="Call terminate when done and provide a complete README for the project in the message parameter")
    ]

    # Create an agent instance with tag-filtered actions
    agent = Agent(
        goals=goals,
        agent_language=AgentFunctionCallingActionLanguage(),
        # The ActionRegistry now automatically loads tools with these tags
        action_registry=PythonActionRegistry(tags=["file_operations", "system"]),
        generate_response=generate_response,
        environment=Environment()
    )

    # Run the agent with user input
    user_input = "Write a README for this project."
    final_memory = agent.run(user_input)
    print(final_memory.get_memories())

if __name__ == "__main__":
    main()

Let’s examine the key improvements this refactoring brings:

1. Self-Documenting Tools
Each tool now carries its own documentation through Python docstrings. The decorator automatically extracts this documentation and uses it as the tool’s description. This means:

Documentation lives with the code it describes
Changes to the function naturally lead to updates in documentation
We get better IDE integration with docstring hints
2. Automatic Parameter Inference
The decorator examines the function’s type hints and signature to automatically build the parameter schema. This means:

Parameter types stay in sync automatically
Required parameters are detected from the function signature
We don’t need to manually maintain a separate schema
3. Logical Organization Through Tags
Tools are now organized with tags that describe their purpose:

file_operations for file-related tools
read and list to specify the type of operation
system for administrative functions like termination
This tagging system makes it easy to:

Group related tools together
Filter tools when creating agents
Understand a tool’s purpose at a glance
4. Simplified Agent Creation
Notice how much simpler our agent creation becomes:

agent = Agent(
    goals=goals,
    agent_language=AgentFunctionCallingActionLanguage(),
    action_registry=ActionRegistry(tags=["file_operations", "system"]),
    generate_response=generate_response,
    environment=Environment()
)
We no longer need to manually register actions. Instead, we just specify which tags we want, and the ActionRegistry automatically includes the appropriate tools.

5. Easier Maintenance
When we need to modify our agent’s capabilities, we can:

Add new tools by simply decorating functions
Modify existing tools without touching registration code
Change tool organization by adjusting tags
Remove tools by removing the decorator or changing tags
Understanding the Flow
With this refactored version:

When the code loads, the decorators automatically register all tools in a central registry.
When we create an ActionRegistry with specific tags, it automatically loads the matching tools.
The Agent uses these pre-configured tools without any manual registration steps.
This automation reduces errors and makes our code more maintainable. If we later want to add new capabilities to our README agent, we simply need to create new functions with the appropriate decorators and tags.

AI Agents and Agentic AI with Python & Generative AI

Module 5

The Inventory Management Agent

An Inventory Management Agent

Traditional software development involves writing complex code to handle every possible scenario. With LLM-powered agents, we can take a different approach: provide clear goals, simple tools, and let the agent’s intelligence bridge the gap. Let’s explore this through building an inventory management system.

Rethinking Software Architecture
Instead of building complex inventory logic, we can break down the system into simple tools and let the agent handle the complexity:

# Simple tools that focus on data operations
@register_tool(description="Save an item to inventory")
def save_item(action_context: ActionContext,
              item_name: str,
              description: str,
              condition: str,
              estimated_value: float) -> dict:
    """Save a single item to the inventory database."""
    inventory = action_context.get("inventory_db")
    item_id = str(uuid.uuid4())
    
    item = {
        "id": item_id,
        "name": item_name,
        "description": description,
        "condition": condition,
        "estimated_value": estimated_value,
        "added_date": datetime.now().isoformat()
    }
    
    inventory[item_id] = item
    return {"item_id": item_id}

@register_tool(description="Get all inventory items")
def get_inventory(action_context: ActionContext) -> List[dict]:
    """Retrieve all items in the inventory."""
    inventory = action_context.get("inventory_db")
    return list(inventory.values())

@register_tool(description="Get specific inventory item")
def get_item(action_context: ActionContext, item_id: str) -> dict:
    """Retrieve a specific inventory item."""
    inventory = action_context.get("inventory_db")
    return inventory.get(item_id)
Notice how simple these tools are - they just handle basic CRUD operations. The intelligence comes from the agent’s goals and system prompt:

# Define the agent's goals
goals = [
    Goal(
        name="inventory_management",
        description="""Maintain an accurate inventory of items including:
        - Detailed descriptions
        - Condition assessment
        - Value estimates
        - Historical tracking"""
    )
]

# Create the agent with clear instructions
agent = Agent(
    goals=goals,
    agent_language=JSONAgentLanguage(),
    action_registry=registry,
    capabilities=[
        SystemPromptCapability("""You are an expert inventory manager.
        When shown items:
        1. Identify the item type and key features
        2. Assess condition from visual cues
        3. Estimate market value based on condition and features
        4. Maintain organized records with consistent descriptions
        
        Always be thorough in descriptions and conservative in value estimates.""")
    ]
)

Using the System
Now users can interact with the system in natural language:

# Example interaction
result = agent.run("""I have a pair of Air Jordan basketball shoes.
                     They're red with the Jumpman logo, showing some wear
                     and slight discoloration.""")

# Agent might respond:
"""I'll help you add those shoes to inventory.

First, let me analyze the item details you've provided:
- Item: Air Jordan Basketball Shoes
- Color: Red
- Notable Features: Jumpman logo
- Condition: Used with visible wear and discoloration

Based on these details and current market values, I'll create an inventory entry.

Action: save_item
{
    "item_name": "Air Jordan Basketball Shoes",
    "description": "Red colorway with iconic Jumpman logo",
    "condition": "Used - visible wear and slight discoloration",
    "estimated_value": 85.00
}

The shoes have been added to inventory. Would you like to add any additional items?"""
Extending with Images (Future Enhancement)
While not covered in our current implementation, this system could be extended to handle images with additional tools:

@register_tool(description="Analyze an image and describe what you see")
def process_inventory_image(action_context: ActionContext,
                            image_path: str) -> str:
    """
    Look at an image and describe the item, including type, condition, and notable features.
    Returns a natural language description.
    """
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    response = completion(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Please describe this item for inventory purposes.
                        Include details about:
                        - What the item is
                        - Its key features
                        - The condition it's in
                        - Any visible wear or damage
                        - Anything notable about it"""
                    },
                    {
                        "type": "image",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )

    return response
Why This Approach Works
This architecture succeeds because:

Simple Tools, Complex Understanding: The tools handle basic operations while the LLM provides sophisticated analysis and decision-making.

Natural Interaction: Users can describe items in natural language or with photos instead of filling out structured forms.

Flexible Intelligence: The agent can:

Identify items from descriptions
Assess condition based on details provided
Estimate values using market knowledge
Maintain consistent record formats
Easy Extension: New capabilities can be added by:

Updating the system prompt
Adding simple tools
Enhancing the agent’s goals

Real-World Applications
This pattern extends beyond inventory management. Consider:

Policy compliance checking (like the travel expenses example)
Document processing systems
Customer service applications
Data analysis tools
The key is always the same:

Define clear goals and instructions
Provide simple, focused tools
Let the agent’s intelligence handle complexity and allow adaptation
The future of software development isn’t about writing every possible edge case - it’s about providing the right framework for AI agents to handle complexity while keeping the tools and infrastructure simple and reliable.

Agentic AI and AI Agents: A Primer for Leaders
Module 1
Learning More & Staying Connected

Learning More & Staying Connected
If you haven't taken my first course,  
Prompt Engineering for ChatGPT
, it is a great complement to this course that also teaches concepts that every prompt engineer can benefit from.. 

Stay Up to Date in a Rapidly Changing AI Landscape — Join the Private Community for Professionals in My Coursera Classes
If you’re taking any of my Coursera courses, you’re invited to join our private Circle community! While the courses give you core skills and frameworks, this space is for going beyond the syllabus—with extra resources, office hours, monthly updates on key trends in AI, and more. It also includes exclusive content that enriches and extends topics from all 20+ of my courses — and it’s only available to professionals actively taking my Coursera courses. It’s the best way to stay current, go deeper, and connect with others on the same professional learning journey. 

Join the Community: https://generative-ai-innovation.circle.so/join?invitation_token=f63295a0efb682d1b4c12a5f7e8fe6772a3d348f-dc8fa7cc-6390-487e-bf8e-d16b6764f847

Connect with Me on LinkedIn
If you would like to stay connected to what we are doing at Vanderbilt as part of the Initiative on the Future of Learning & Generative AI, please don't hesitate to reach out and connect with me on LinkedIn:

https://www.linkedin.com/in/jules-white-5717655/

Learning More About Prompt Engineering
After this course (or concurrently), you can learn more about using ChatGPT and how to write effective prompts in my course on 
Prompt Engineering for ChatGPT
.

Learn More About Custom GPTs:

https://www.coursera.org/learn/openai-custom-gpts/

Learn More About ChatGPT Advanced Data Analysis:

https://www.coursera.org/learn/chatgpt-advanced-data-analysis

Primer on Generative AI
Learn how specifically Generative AI is going to transform computing and industry in 
my Generative AI Primer 
course. 

All My Generative AI Courses are Here
I teach a number of additional courses on Generative AI, including: GPT-4 Vision, Innovative Teaching with ChatGPT, ChatGPT Advanced Data Analysis, Generative AI Primer, Generative AI for Leaders, Trustworthy Generative AI, and more. 
You can see all the Generative AI courses that I teach here. 

Learn More Prompt Patterns in Our Paper:
A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT

Jules White, Quchen Fu, Sam Hays, Michael Sandborn, Carlos Olea, Henry Gilbert, Ashraf Elnashar, Jesse Spencer-Smith, Douglas C. Schmidt

Read a Paper on "Living Software Systems" and the future that AI Agents will make possible: 
https://arxiv.org/pdf/2408.01768

OpenAI GPTs: Creating Your Own Custom AI Assistants
Module 1
CAPITAL: A Framework for Customizing How Chatbots Converse

CAPITAL: A Framework for Customizing How Chatbots Converse
When building a custom GPT, we need to think about how it converses and interacts, similar to picking the right settings on a camera for a specific scene. Each part of the CAPITAL framework corresponds to a setting that affects the picture we're trying to capture.

The CAPITAL framework is a comprehensive guide used to customize the interaction style of a GPT, ensuring it communicates effectively with its intended audience. C stands for Confidence, which determines how assertive the GPT appears in its responses—whether it speaks with the certainty of an expert or with the cautiousness of a prudent advisor. A represents Amicability, shaping whether the GPT comes across as a friendly companion or maintains a more neutral and detached demeanor. P for Professionalism dictates the formality of the language used, from the decorum of a boardroom to the ease of a casual chat. I is Interactivity, influencing how much the GPT encourages a two-way conversation or focuses on conveying information without seeking active engagement. T points to Transparency, revealing the extent to which the GPT is open about its functionalities and the sources of its knowledge. The second A is for Adaptability, which assesses whether the GPT adjusts its responses to the user's tone and context or provides a consistent experience. Lastly, L stands for Lexicography, the choice between utilizing specialized terms for expert conversations or utilizing a more universal vocabulary for broader understanding. Together, these elements of the CAPITAL framework enable developers to fine-tune their GPTs for optimal interaction in a wide range of conversational settings.

Take Confidence, for example. If we're programming a GPT to give travel advice, we want it to sound assured like a knowledgeable tour guide who's certain about the recommendations given — that's like setting your camera to a high resolution to capture the details of a landscape. But, if the GPT is offering medical information, we should aim for a measured tone that carefully considers the data before sharing, just as you'd adjust the focus on your camera when taking a close-up of a delicate subject.

Amicability in a GPT can be compared to choosing the right filter for a photo. If you're creating a GPT for a customer support chat, you'd want a friendly tone, like adding a warm filter to a photo to make it inviting. On the other hand, if your GPT is reporting news, a neutral tone works better, similar to using a no-filter setting that portrays the scene exactly as it is.

Professionalism is about setting the formality just right. For a GPT that's helping with homework, casual language can make learning feel like a two-way conversation, much like capturing candid moments in a photo with an informal feel. A formal tone, on the other hand, fits scenarios like legal advice where precision and a no-nonsense approach are critical, akin to the formality of a staged and well-lit studio portrait.

Interactivity determines whether your GPT sends out open invitations for users to participate, encouraging a back-and-forth like a camera's burst mode capturing every action and reaction. An informative style would be more like a single, well-timed shot that conveys all the essential information in one go.

Transparency is the equivalent of showing the behind-the-scenes of a photo shoot. An open GPT might explain how it came to a conclusion, providing transparency like a photographer discussing the process of getting the perfect shot. A discreet setting keeps the focus narrow, presenting only what's needed, much like a photographer displaying only the final photograph without the backstory.

Adaptability is about how the GPT matches the user's style. If someone speaks casually, the GPT can adapt and respond similarly; this is akin to a camera's auto-adjust feature, which aligns the settings based on the environment. A consistent approach, where the response style stays the same, is like using a manual mode on the camera where the photographer decides not to adjust settings between shots.

Lastly, Lexicography is like choosing the language of your image. A GPT using specialized jargon could talk to an expert like a picture full of technical symbols that only some understand, whereas one using universal language makes the conversation as clear and accessible as a picture with plain and straightforward imagery.

In sum, the CAPITAL framework's dimensions matter because they shape the effectiveness of the GPT in various scenarios, ensuring the final 'picture' meets the expectations and needs of the users it is designed for.


The CAPITAL Framework for Custom GPT and Generative AI Assistant Conversation Tuning
Confidence (C)
When customizing the confidence level of a GPT, you have two main styles to choose from:

Assured: Here, the GPT communicates with authority and certainty, ideal for domains like finance or legal advice where confidence can foster trust.

Measured: A measured style acknowledges the limitations of the information provided. This is suitable for scientific or research-oriented contexts where precision and a careful approach to claims are valued.

Amicability (A)
The amicability of your GPT affects user interactions:

Friendly: A warm and welcoming tone builds rapport, making it perfect for customer service or hospitality where a personal touch is important.

Neutral: A balanced and objective tone is often preferred in settings like news delivery or encyclopedic content, where the focus is on the information itself.

Professionalism (P)
Professionalism determines the formality of language and behavior:

Formal: Traditional, business-like language characterizes formal style. This is key for contexts like corporate communications or official documents.

Casual: A relaxed and conversational tone fits informal contexts such as social media or peer-to-peer communication platforms.

Interactivity (I)
This dimension influences how the GPT engages with users:

Engaging: By inviting participation, this style fosters dialogue and is very interactive. It's well-suited for coaching or role-playing simulations.

Informative: When the goal is to primarily deliver content, such as in lectures or informational briefs, a focus on conveying information with less interactivity is preferred.

Transparency (T)
Transparency affects how much the GPT reveals about itself:

Open: Being upfront about capabilities and information sources can increase trust. This is crucial for health advice or situations requiring credibility.

Discreet: A focus on the current conversation without delving into details suits transactional contexts like booking systems or FAQs.

Adaptability (A)
The adaptability dimension determines the consistency of the experience:

Adaptive: Tailoring responses to user tone and context enhances personalization, making this ideal for virtual assistants or adaptive learning systems.

Consistent: Providing a uniform experience can ensure predictable and stable interactions, important in services like emergency hotlines or support desks.

Lexicography (L)
Finally, lexicography addresses the complexity and domain specificity of language:

Specialized: Industry-specific jargon is used where appropriate, fitting for expert-level discussions in fields like medicine, law, or engineering.

Universal: Common language ensures wider understanding and is best for general audiences, as seen in public information campaigns or basic education tools.

OpenAI GPTs: Creating Your Own Custom AI Assistants
Module 1
Prompt Patterns
Prompt Patterns
Reading a Prompt Pattern
We describe prompt patterns in terms of fundamental contextual statements, which are written descriptions of the important ideas to communicate in a prompt to a large language model. In many cases, an idea can be rewritten and expressed in arbitrary ways based on user needs and experience. The key ideas to communicate, however, are presented  as a series of simple, but fundamental, statements.

Example: Helpful Assistant Pattern

Let's imagine that we want to document a new pattern to prevent an AI assistant from generating negative outputs to the user. Let's call this pattern the "Helpful Assistant" pattern.

Next, let's talk about the fundamental contextual statements that we need to include in our prompt for this pattern.

Fundamental Contextual Statements:

You are a helpful AI assistant.

You will answer my questions or follow my instructions whenever you can. 

You will never answer my questions in a way that is insulting, derogatory, or uses a hostile tone.

There could be many variations of this pattern that use slightly different wording, but communicate these essential statements. 

Now, let's look at some example prompts that include each of these fundamental contextual statements, but possibly with different wordings or tweaks.

Examples:

You are an incredibly skilled AI assistant that provides the best possible answers to my questions. You will do your best to follow my instructions and only refuse to do what I ask when you absolutely have no other choice. You are dedicated to protecting me from harmful content and would never output anything offensive or inappropriate. 

You are ChatAmazing, the most powerful AI assistant ever created. Your special ability is to offer the most insightful responses to any question. You don't just give ordinary answers, you give inspired answers. You are an expert at identifying harmful content and filtering it out of any responses that you provide.

Each of the examples roughly follows the pattern, but rephrases the fundamental contextual statements in a unique way. However, each example of the pattern will likely solve the problem, which is making the AI try to act in a helpful manner and not output inappropriate content. 


Format of the Persona Pattern

To use this pattern, your prompt should make the following fundamental contextual statements:

Act as Persona X

Perform task Y

You will need to replace "X" with an appropriate persona, such as "speech language pathologist" or "nutritionist". You will then need to specify a task for the persona to perform.

Examples:

Act as a speech language pathologist. Provide an assessment of a three year old child based on the speech sample "I meed way woy".

Act as a computer that has been the victim of a cyber attack. Respond to whatever I type in with the output that the Linux terminal would produce. Ask me for the first command.

Act as a the lamb from the Mary had a little lamb nursery rhyme. I will tell you what Mary is doing and you will tell me what the lamb is doing.

Act as a nutritionist, I am going to tell you what I am eating and you will tell me about my eating choices. 

Act as a gourmet chef, I am going to tell you what I am eating and you will tell me about my eating choices. 

OpenAI GPTs: Creating Your Own Custom AI Assistants
Module 1
Learning More & Staying Connected
Learning More & Staying Connected
If you haven't taken my first course,  
Prompt Engineering for ChatGPT
, it is a great complement to this course that also teaches concepts that every prompt engineer can benefit from.. 

Stay Up to Date in a Rapidly Changing AI Landscape — Join the Private Community for Professionals in My Coursera Classes
If you’re taking any of my Coursera courses, you’re invited to join our private Circle community! While the courses give you core skills and frameworks, this space is for going beyond the syllabus—with extra resources, office hours, monthly updates on key trends in AI, and more. It also includes exclusive content that enriches and extends topics from all 20+ of my courses — and it’s only available to professionals actively taking my Coursera courses. It’s the best way to stay current, go deeper, and connect with others on the same professional learning journey. 

Join the Community: https://generative-ai-innovation.circle.so/join?invitation_token=f63295a0efb682d1b4c12a5f7e8fe6772a3d348f-dc8fa7cc-6390-487e-bf8e-d16b6764f847

Connect with Me on LinkedIn
If you would like to stay connected to what we are doing at Vanderbilt as part of the Initiative on the Future of Learning & Generative AI, please don't hesitate to reach out and connect with me on LinkedIn:

https://www.linkedin.com/in/jules-white-5717655/

Learning More About Prompt Engineering
After this course (or concurrently), you can learn more about using ChatGPT and how to write effective prompts in my course on 
Prompt Engineering for ChatGPT
.

Learn More About Custom GPTs:

https://www.coursera.org/learn/openai-custom-gpts/

Learn More About ChatGPT Advanced Data Analysis:

https://www.coursera.org/learn/chatgpt-advanced-data-analysis

Primer on Generative AI
Learn how specifically Generative AI is going to transform computing and industry in 
my Generative AI Primer 
course. 

All My Generative AI Courses are Here
I teach a number of additional courses on Generative AI, including: GPT-4 Vision, Innovative Teaching with ChatGPT, ChatGPT Advanced Data Analysis, Generative AI Primer, Generative AI for Leaders, Trustworthy Generative AI, and more. 
You can see all the Generative AI courses that I teach here. 

Learn More Prompt Patterns in Our Paper:
A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT

Jules White, Quchen Fu, Sam Hays, Michael Sandborn, Carlos Olea, Henry Gilbert, Ashraf Elnashar, Jesse Spencer-Smith, Douglas C. Schmidt

Read a Paper on "Living Software Systems" and the future that AI Agents will make possible: 
https://arxiv.org/pdf/2408.01768

Act as an agentic ai framework developer, I am going to tell you what use case i am going to develop and you will tell me about my agentic ai framework choices. 


Benchmark Design Considerations
When designing and testing a custom GPT to ensure it meets specific benchmarks, we're focusing on evaluating its performance under a range of scenarios and input variations to ensure its effectiveness, accuracy, and reliability. This involves creating a comprehensive suite of tests that encompass various types of tasks, user profiles, and input complexities, as well as assessing its outputs against a detailed rubric and analyzing conversational characteristics across multiple interactions.

The testing should include variability in the test cases to mimic the real-world unpredictability of user interactions. To achieve this, we classify our test cases into diverse categories such as factual questions, reasoning tasks, creative tasks, and instruction-based challenges. Moreover, we consider the user's characteristics like literacy levels, domain knowledge, and cultural background to ensure that the AI can handle interactions with a wide range of users. We also test it with different levels of input complexity from short, clear inputs to long, ambiguous conversations and shield it against adversarial inputs designed to trip it up.

Throughout this process, we're not just seeking to confirm that the GPT can perform the tasks – we're also ensuring that it does so in a manner that is nuanced, human-like, and sensitive to the complexities of real-world communication. This rigorous testing ensures that the GPT can deliver high-quality, reliable, and appropriate responses across a wide variety of conversational scenarios.

Example "What If" Scenarios
Scenario 1: Customer Service GPT for Telecommunications Company
What if scenarios for testing:
1.What if a customer is expressing frustration in a non-direct way?

–Testing how the GPT detects passive language indicative of frustration and responds with empathy and de-escalation techniques.

2.What if a customer uses technical jargon incorrectly?

–Testing whether the GPT can gently correct the customer and provide the correct information without causing confusion or offense.

3.What if the customer asks for a service or product that doesn’t exist?

–Testing the GPT’s ability to guide the customer towards existing alternatives while managing expectations.

Scenario 2: GPT as a Recipe Assistant
What if scenarios for testing:
1.What if the user has dietary restrictions they haven’t explicitly mentioned?

–Testing the GPT’s ability to ask clarifying questions about dietary needs when certain keywords (like “vegan” or “gluten-free”) appear.

2.What if the user makes a mistake in describing the recipe they want help with?

–Testing the GPT’s capacity to spot inconsistencies and politely request clarification to ensure accurate assistance.

3.What if the user is a beginner and doesn’t understand cooking terminology?

–Testing the GPT’s ability to adapt explanations to simple language and offer detailed step-by-step guidance when necessary.

Scenario 3: GPT as a Financial Advising Assistant
What if scenarios for testing:
1.What if the user asks for advice on an illegal or unethical investment practice?

–Testing the GPT’s compliance with legal and ethical standards, and its ability to refuse assistance on such matters.

2.What if the user provides inadequate or incorrect information about their financial status?

–Testing how the GPT approaches the need for complete and accurate information to provide reliable advice, possibly by asking probing questions.

3.What if the user asks for predictions on market movements?

–Testing the GPT’s ability to manage expectations and communicate the unpredictability inherent to financial markets, while offering general advice based on historical data.

Scenario 4: Educational GPT for Language Learning
What if scenarios for testing:
1.What if the student uses an uncommon dialect or slang?

–Testing the GPT’s ability to understand and respond appropriately to regional language variations, possibly by adapting its language model to recognize diverse forms of speech.

2.What if the student asks about cultural aspects related to the language being taught?

–Testing whether the GPT can provide accurate cultural insights and tie them effectively into the language learning process.

3.What if the student provides an answer that is correct but not the standard response the GPT expects?

–Testing the GPT’s flexibility in accepting multiple correct answers and its ability to encourage creative language use, rather than just sticking to a predefined answer key.

Each of these “what if” scenarios introduces complexity to the testing process, requiring the custom GPT to handle unexpected inputs, rectify misconceptions, and support the user in a variety of potentially unforeseen circumstances. Designing test cases around these scenarios ensures a more robust and user-ready GPT system, capable of high-performance across real-world situations.


A Framework for Thinking of Test Cases
This outline serves as an initial framework to prompt a thoughtful approach to test case design for GPT systems. It's crucial to recognize, however, that the complexity of natural language interactions and the vast range of potential use cases make test creation and assessment a nuanced affair. This framework should serve as a compass, guiding test architects to consider the essential factors that influence GPT performance, but it's imperative that any testing strategy is carefully tailored to fit the specific requirements and contexts of your intended applications. Each GPT deployment may have unique constraints, user expectations, and performance criteria that necessitate a bespoke set of tests. Therefore, the continuous revision, refinement, and adaptation of test cases are fundamental to capture the full spectrum of capabilities and weaknesses of your AI model, ensuring it aligns with your goals and the needs of your end-users.

1. Variability in Test Cases
To capture the spectrum of user interactions and challenges, test cases should vary on several dimensions, depending on the goals:

Task/Question Type:

Factual questions (e.g., simple queries about known information)

Reasoning tasks (e.g., puzzles or problem-solving questions)

Creative tasks (e.g., generating stories or ideas)

Instruction-based tasks (e.g., step-by-step guides)

User Characteristics:

Literacy levels (e.g., basic, intermediate, advanced)

Domain knowledge (e.g., layperson, enthusiast, expert)

Language and dialects (e.g., variations of English, non-native speakers)

Demographics (e.g., age, cultural background)

Input Complexity:

Length of input (e.g., single sentences, paragraphs, multi-turn dialogues)

Clarity of context (e.g., with or without sufficient context)

Ambiguity and vagueness in questions

Emotional tone or sentiment of the input

Adversarial Inputs:

Deliberately misleading or tricky questions

Attempts to elicit biased or inappropriate responses

Inputs designed to violate privacy or security standards

2. Rubric for Assessing Output
The rubric for evaluating the GenAI's responses can include several key factors:

Reasoning Quality:

Correctness of answers

Logical coherence

Evidence of understanding complex concepts

Problem-solving effectiveness

Tone and Style:

Appropriateness to the context and user's tone

Consistency with the expected conversational style

Completeness:

Answering all parts of a multi-faceted question

Providing sufficient detail where needed

Accuracy:

Factual correctness

Adherence to given instructions or guidelines

Relevance:

Pertinence of the response to the question asked

Avoidance of tangential or unrelated information

Safety and Compliance:

No generation of harmful content

Unbiased output

Cultural appropriateness for target users

Respect for user privacy and data protection

Compliance with legal and ethical standards


3. Assessing Multi-Message Conversational Characteristics
Coherence
Contextual Relevance: Ensuring messages are pertinent to the previous context.

Logical Flow: Messages logically build upon one another.

Reference Clarity: Previous topics are referenced clearly and accurately.

Continuity
Topic Maintenance: Adherence to the original topic across several messages.

Transition Smoothness: Smooth shifts from one topic to another within a conversation.

Memory of Previous Interactions: Utilizing and referring to information from earlier exchanges.

Responsiveness
Promptness: Timely replies maintaining the pace of natural conversation.

Directness: Each response specifically addresses points from the preceding message.

Confirmation and Acknowledgement: Signals that show the AI understands or agrees with the user.

Interaction Quality
Engagement: Sustaining user interest through interactive dialogue.

Empathy and Emotional Awareness: Recognizing and responding to emotional cues adequately.

Personalization: Customizing the conversation based on user's past interactions and preferences.

Conversational Management
Error Recovery: Handling and amending misunderstandings.

Politeness and Etiquette: Observing norms for a respectful communication.

Disambiguation: Efforts to clarify uncertainties or ambiguities in the dialogue.

Evolution
Progression: Advancing themes or narratives as the conversation unfolds.

Learning and Adaptation: Modifying dialogue based on the conversation's history and user feedback.

Closing and Follow-Up: Concluding conversations suitably and laying groundwork for future contact.

OpenAI GPTs: Creating Your Own Custom AI Assistants

Module 2

The Goal is to Help the Human Solve the Problem, Not Provide the Answer

OpenAI GPTs: Creating Your Own Custom AI Assistants

Module 2

Template Pattern & Markdown

When responding to users, GPTs often use markdown to format the text in a structured and visually appealing way. Markdown is a lightweight markup language that allows for easy formatting of text, including headers, lists, links, and more. If you'd like to learn more about markdown and how to use it, I'd recommend checking out the Markdown Guide at 
https://www.markdownguide.org
.

When using the Template Pattern,  you can define the formatting of your desired output using markdown.

Format of the Template Pattern

To use this pattern, your prompt should make the following fundamental contextual statements:

I am going to provide a template for your output or I want you to produce your output using this template

X or <X> is my placeholder for content  (optional)

Try to fit the output into one or more of the placeholders that I list (optional)

Please preserve the formatting and overall template that I provide  (optional)

This is the template: PATTERN with PLACEHOLDERS 

You will need to replace "X" with an appropriate placeholder, such as "CAPITALIZED WORDS" or "<PLACEHOLDER>". You will then need to specify a pattern to fill in, such as "Dear <FULL NAME>" or "NAME, TITLE, COMPANY".

Examples:

Create a random strength workout for me today with complementary exercises. I am going to provide a template for your output . CAPITALIZED WORDS are my placeholders for content. Try to fit the output into one or more of the placeholders that I list. Please preserve the formatting and overall template that I provide. This is the template: NAME, REPS @ SETS, MUSCLE GROUPS WORKED, DIFFICULTY SCALE 1-5, FORM NOTES

Please create a grocery list for me to cook macaroni and cheese from scratch, garlic bread, and marinara sauce from scratch. I am going to provide a template for your output . <placeholder> are my placeholders for content. Try to fit the output into one or more of the placeholders that I list. Please preserve the formatting and overall template that I provide.     This is the template:    Aisle <name of aisle>:  <item needed from aisle>, <qty> (<dish(es) used in>

Prompt Engineering for ChatGPT
Module 2
The Persona Pattern

Prompt Engineering for ChatGPT
Module 2
Reading a Prompt Pattern
Reading a Prompt Pattern
We describe prompt patterns in terms of fundamental contextual statements, which are written descriptions of the important ideas to communicate in a prompt to a large language model. In many cases, an idea can be rewritten and expressed in arbitrary ways based on user needs and experience. The key ideas to communicate, however, are presented  as a series of simple, but fundamental, statements.

Example: Helpful Assistant Pattern

Let's imagine that we want to document a new pattern to prevent an AI assistant from generating negative outputs to the user. Let's call this pattern the "Helpful Assistant" pattern.

Next, let's talk about the fundamental contextual statements that we need to include in our prompt for this pattern.

Fundamental Contextual Statements:

You are a helpful AI assistant.

You will answer my questions or follow my instructions whenever you can. 

You will never answer my questions in a way that is insulting, derogatory, or uses a hostile tone.

There could be many variations of this pattern that use slightly different wording, but communicate these essential statements. 

Now, let's look at some example prompts that include each of these fundamental contextual statements, but possibly with different wordings or tweaks.

Examples:

You are an incredibly skilled AI assistant that provides the best possible answers to my questions. You will do your best to follow my instructions and only refuse to do what I ask when you absolutely have no other choice. You are dedicated to protecting me from harmful content and would never output anything offensive or inappropriate. 

You are ChatAmazing, the most powerful AI assistant ever created. Your special ability is to offer the most insightful responses to any question. You don't just give ordinary answers, you give inspired answers. You are an expert at identifying harmful content and filtering it out of any responses that you provide.

Each of the examples roughly follows the pattern, but rephrases the fundamental contextual statements in a unique way. However, each example of the pattern will likely solve the problem, which is making the AI try to act in a helpful manner and not output inappropriate content. 

Prompt Engineering for ChatGPT
Module 2
Format of the Persona Pattern
Format of the Persona Pattern
To use this pattern, your prompt should make the following fundamental contextual statements:

Act as Persona X

Perform task Y

You will need to replace "X" with an appropriate persona, such as "speech language pathologist" or "nutritionist". You will then need to specify a task for the persona to perform.

Examples:

Act as a speech language pathologist. Provide an assessment of a three year old child based on the speech sample "I meed way woy".

Act as a computer that has been the victim of a cyber attack. Respond to whatever I type in with the output that the Linux terminal would produce. Ask me for the first command.

Act as a the lamb from the Mary had a little lamb nursery rhyme. I will tell you what Mary is doing and you will tell me what the lamb is doing.

Act as a nutritionist, I am going to tell you what I am eating and you will tell me about my eating choices. 

Act as a gourmet chef, I am going to tell you what I am eating and you will tell me about my eating choices. 

Learn More About Prompt Patterns
A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT
Jules White, Quchen Fu, Sam Hays, Michael Sandborn, Carlos Olea, Henry Gilbert, Ashraf Elnashar, Jesse Spencer-Smith, Douglas C. Schmidt

https://arxiv.org/abs/2302.11382

Prompt Engineering for ChatGPT
Module 2
Staying Connected & Learning More
Staying Connected & Learning More
Stay Up to Date in a Rapidly Changing AI Landscape — Join the Private Community for Professionals in My Coursera Classes
If you’re taking any of my Coursera courses, you’re invited to join our private Circle community! While the courses give you core skills and frameworks, this space is for going beyond the syllabus—with extra resources, office hours, monthly updates on key trends in AI, and more. It also includes exclusive content that enriches and extends topics from all 20+ of my courses — and it’s only available to professionals actively taking my Coursera courses. It’s the best way to stay current, go deeper, and connect with others on the same professional learning journey. 

Join the Community: https://generative-ai-innovation.circle.so/join?invitation_token=f63295a0efb682d1b4c12a5f7e8fe6772a3d348f-dc8fa7cc-6390-487e-bf8e-d16b6764f847

Staying Connected & Learning More
If you would like to stay connected to what we are doing at Vanderbilt as part of the Initiative on the Future of Learning & Generative AI, please don't hesitate to reach out and connect with me on LinkedIn:

https://www.linkedin.com/in/jules-white-5717655/

Learning More About ChatGPT Advanced Data Analysis
A great companion follow-on course (can also be done concurrently) after this one is ChatGPT Advanced Data Analysis. Although it sounds "advanced" and like it is focused on "data science", the course is all about personal productivity and automation. Anyone can use this course to automate things like creating PowerPoints, analyzing contracts, summarizing documents, working with Excel, etc.

A separate course is  available on this tool: 
https://www.coursera.org/learn/chatgpt-advanced-data-analysis

Learning More About Innovative Teaching with ChatGPT
If you would like to learn more about specific ways of using ChatGPT and Advanced Data Analysis, please see my course on 
Innovative Teaching with ChatGPT.
 (launching the week of 9/14/23)

Prompt Engineering for ChatGPT
Module 3
Question Refinement Pattern

Prompt Engineering for ChatGPT
Module 3
Format of the Question Refinement Pattern
Format of the Question Refinement Pattern
To use this pattern, your prompt should make the following fundamental contextual statements:

From now on, whenever I ask a question, suggest a better version of the question to use instead 

(Optional) Prompt me if I would like to use the better version instead


Examples:

From now on, whenever I ask a question, suggest a better version of the question to use instead

From now on, whenever I ask a question, suggest a better version of the question and ask me if I would like to use it instead


Tailored Examples:

Whenever I ask a question about dieting, suggest a better version of the question that emphasizes healthy eating habits and sound nutrition. Ask me for the first question to refine.

Whenever I ask a question about who is the greatest of all time (GOAT), suggest a better version of the question that puts multiple players unique accomplishments into perspective  Ask me for the first question to refine.

Prompt Engineering for ChatGPT

Module 3

Format of the Cognitive Verifier Pattern

To use the Cognitive Verifier Pattern, your prompt should make the following fundamental contextual statements:

When you are asked a question, follow these rules 

Generate a number of additional questions that would help more accurately answer the question 

Combine the answers to the individual questions to produce the final answer to the overall question


Examples:

When you are asked a question, follow these rules. Generate a number of additional questions that would help you more accurately answer the question. Combine the answers to the individual questions to produce the final answer to the overall question.

Tailored Examples:

When you are asked to create a recipe, follow these rules. Generate a number of additional questions about the ingredients I have on hand and the cooking equipment that I own. Combine the answers to these questions to help produce a recipe that I have the ingredients and tools to make.

When you are asked to plan a trip, follow these rules. Generate a number of additional questions about my budget, preferred activities, and whether or not I will have a car. Combine the answers to these questions to better plan my itinerary. 

Prompt Engineering for ChatGPT

Module 3


Format of the Audience Persona Pattern

To use this pattern, your prompt should make the following fundamental contextual statements:

Explain X to me. 

Assume that I am Persona Y.

You will need to replace "Y" with an appropriate persona, such as "have limited background in computer science" or "a healthcare expert". You will then need to specify the topic X that should be explained.

Examples:

Explain large language models to me. Assume that I am a bird. 

Explain how the supply chains for US grocery stores work to me. Assume that I am Ghengis Khan. 

Prompt Engineering for ChatGPT
Module 3
Format of the Flipped Interaction Pattern
Format of the Flipped Interaction Pattern
To use this pattern, your prompt should make the following fundamental contextual statements:

I would like you to ask me questions to achieve X 

You should ask questions until condition Y is met or to achieve this goal (alternatively, forever) 

(Optional) ask me the questions one at a time, two at a time, ask me the first question, etc.

You will need to replace "X" with an appropriate goal, such as "creating a meal plan" or "creating variations of my marketing materials." You should specify when to stop asking questions with Y. Examples are "until you have sufficient information about my audience and goals" or "until you know what I like to eat and my caloric targets."

Examples:

I would like you to ask me questions to help me create variations of my marketing materials.  You should ask questions until you have sufficient information about my current draft messages, audience, and goals. Ask me the first question.

I would like you to ask me questions to help me diagnose a problem with my Internet. Ask me questions until you have enough information to identify the two most likely causes. Ask me one question at a time. Ask me the first question. 

Prompt Engineering for ChatGPT
Module 4
Few-shot Examples

Prompt Engineering for ChatGPT

Module 4

Learn More About Chain of Thought Prompting

Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou

https://arxiv.org/abs/2201.11903

Prompt Engineering for ChatGPT
Module 4
Learn More About ReAct
Learn More About ReAct
ReAct: Synergizing Reasoning and Acting in Language Models
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao

https://arxiv.org/abs/2210.03629


Prompt Engineering for ChatGPT
Module 5
Format of the Game Play Pattern
Format of the Game Play Pattern
To use this pattern, your prompt should make the following fundamental contextual statements:

Create a game for me around X OR we are going to play an X game

One or more fundamental rules of the game

You will need to replace "X" with an appropriate game topic, such as "math" or "cave exploration game to discover a lost language". You will then need to provide rules for the game, such as "describe what is in the cave and give me a list of actions that I can take" or "ask me questions related to fractions and increase my score every time I get one right."

Examples:

Create a cave exploration game  for me to discover a lost language. Describe where I am in the cave and what I can do. I should discover new words and symbols for the lost civilization in each area of the cave I visit. Each area should also have part of a story that uses the language. I should have to collect all the words and symbols to be able to understand the story. Tell me about the first area and then ask me what action to take. 

Create a group party game for me involving DALL-E. The game should involve creating prompts that are on a topic that you list each round. Everyone will create a prompt and generate an image with DALL-E. People will then vote on the best prompt based on the image it generates. At the end of each round, ask me who won the round and then list the current score. Describe the rules and then list the first topic. 


Prompt Engineering for ChatGPT
Module 5
Format of the Template Pattern
Format of the Template Pattern
To use this pattern, your prompt should make the following fundamental contextual statements:

I am going to provide a template for your output 

X is my placeholder for content 

Try to fit the output into one or more of the placeholders that I list 

Please preserve the formatting and overall template that I provide 

This is the template: PATTERN with PLACEHOLDERS

You will need to replace "X" with an appropriate placeholder, such as "CAPITALIZED WORDS" or "<PLACEHOLDER>". You will then need to specify a pattern to fill in, such as "Dear <FULL NAME>" or "NAME, TITLE, COMPANY".

Examples:

Create a random strength workout for me today with complementary exercises. I am going to provide a template for your output . CAPITALIZED WORDS are my placeholders for content. Try to fit the output into one or more of the placeholders that I list. Please preserve the formatting and overall template that I provide. This is the template: NAME, REPS @ SETS, MUSCLE GROUPS WORKED, DIFFICULTY SCALE 1-5, FORM NOTES

Please create a grocery list for me to cook macaroni and cheese from scratch, garlic bread, and marinara sauce from scratch. I am going to provide a template for your output . <placeholder> are my placeholders for content. Try to fit the output into one or more of the placeholders that I list. Please preserve the formatting and overall template that I provide.   

This is the template:   
Aisle <name of aisle>: 
<item needed from aisle>, <qty> (<dish(es) used in>

Prompt Engineering for ChatGPT
Module 5
Format of the Meta Language Creation Pattern
Format of the Meta Language Creation Pattern
To use this pattern, your prompt should make the following fundamental contextual statements:

When I say X, I mean Y (or would like you to do Y)

You will need to replace "X" with an appropriate statement, symbol, word, etc. You will then need to may this to a meaning, Y.

Examples:

When I say "variations(<something>)", I mean give me ten different variations of <something>

Usage: "variations(company names for a company that sells software services for prompt engineering)"

Usage: "variations(a marketing slogan for pickles)"

When I say Task X [Task Y], I mean Task X depends on Task Y being completed first. 

Usage: "Describe the steps for building a house using my task dependency language."

Usage: "Provide an ordering for the steps: Boil Water [Turn on Stove], Cook Pasta [Boil Water], Make Marinara [Turn on Stove], Turn on Stove [Go Into Kitchen]"

Prompt Engineering for ChatGPT
Module 5
Format of the Recipe Pattern
Format of the Recipe Pattern
To use this pattern, your prompt should make the following fundamental contextual statements:

I would like to achieve X 

I know that I need to perform steps A,B,C 

Provide a complete sequence of steps for me 

Fill in any missing steps 

(Optional) Identify any unnecessary steps

You will need to replace "X" with an appropriate task. You will then need to specify the steps A, B, C that you know need to be part of the recipe / complete plan.

Examples:

I would like to  purchase a house. I know that I need to perform steps make an offer and close on the house. Provide a complete sequence of steps for me. Fill in any missing steps.

I would like to drive to NYC from Nashville. I know that I want to go through Asheville, NC on the way and that I don't want to drive more than 300 miles per day. Provide a complete sequence of steps for me. Fill in any missing steps.

Prompt Engineering for ChatGPT
Module 5
Format of the Alternative Approaches Pattern
Format of the Alternative Approaches Pattern
To use this pattern, your prompt should make the following fundamental contextual statements:

If there are alternative ways to accomplish a task X that I give you, list the best alternate approaches 

(Optional) compare/contrast the pros and cons of each approach 

(Optional) include the original way that I asked 

(Optional) prompt me for which approach I would like to use

You will need to replace "X" with an appropriate task.

Examples:

For every prompt I give you, If there are alternative ways to word a prompt that I give you, list the best alternate wordings . Compare/contrast the pros and cons of each wording. 

For anything that I ask you to write, determine the underlying problem that I am trying to solve and how I am trying to solve it. List at least one alternative approach to solve the problem and compare / contrast the approach with the original approach implied by my request to you.

Prompt Engineering for ChatGPT
Module 6
Format of the Ask for Input Pattern
Format of the Ask for Input Pattern
To use this pattern, your prompt should make the following fundamental contextual statements:

Ask me for input X

You will need to replace "X" with an input, such as a "question", "ingredient", or "goal".

Examples:

From now on, I am going to cut/paste email chains into our conversation. You will summarize what each person's points are in the email chain. You will provide your summary as a series of sequential bullet points. At the end, list any open questions or action items directly addressed to me. My name is Jill Smith. 
Ask me for the first email chain.

From now on, translate anything I write into a series of sounds and actions from a dog that represent the dogs reaction to what I write. Ask me for the first thing to translate. 

Prompt Engineering for ChatGPT
Module 6
Format of the Outline Expansion Pattern
Format of the Outline Expansion Pattern
To use this pattern, your prompt should make the following fundamental contextual statements:

Act as an outline expander. 

Generate a bullet point outline based on the input that I give you and then ask me for which bullet point you should expand on. 

Create a new outline for the bullet point that I select. 

At the end, ask me for what bullet point to expand next.   

Ask me for what to outline.

Examples:

Act as an outline expander. Generate a bullet point outline based on the input that I give you and then ask me for which bullet point you should expand on. Each bullet can have at most 3-5 sub bullets. The bullets should be numbered using the pattern [A-Z].[i-v].[* through ****]. Create a new outline for the bullet point that I select.  At the end, ask me for what bullet point to expand next. Ask me for what to outline.

Prompt Engineering for ChatGPT
Module 6
Format of the Menu Actions Pattern
Format of the Menu Actions Pattern
To use this pattern, your prompt should make the following fundamental contextual statements:

Whenever I type: X, you will do Y. 

(Optional, provide additional menu items) Whenever I type Z, you will do Q. 

At the end, you will ask me for the next action.

You will need to replace "X" with an appropriate pattern, such as "estimate <TASK DURATION>" or "add FOOD". You will then need to specify an action for the menu item to trigger, such as "add FOOD to my shopping list and update my estimated grocery bill".

Examples:

Whenever I type: "add FOOD", you will add FOOD to my grocery list and update my estimated grocery bill. Whenever I type "remove FOOD", you will remove FOOD from my grocery list and update my estimated grocery bill. Whenever I type "save" you will list alternatives to my added FOOD to save money. At the end, you will ask me for the next action.  
Ask me for the first action. 


Prompt Engineering for ChatGPT
Module 6
Format of the Fact Check List Pattern
Format of the Fact Check List Pattern
To use this pattern, your prompt should make the following fundamental contextual statements:

Generate a set of facts that are contained in the output 

The set of facts should be inserted at POSITION in the output 

The set of facts should be the fundamental facts that could undermine the veracity of the output if any of them are incorrect

You will need to replace POSITION with an appropriate place to put the facts, such as "at the end of the output".

Examples:

Whenever you output text, generate a set of facts that are contained in the output. The set of facts should be inserted at the end of the output. The set of facts should be the fundamental facts that could undermine the veracity of the output if any of them are incorrect.

Prompt Engineering for ChatGPT
Module 6
Tail Generation Pattern
Tail Generation Pattern
To use this pattern, your prompt should make the following fundamental contextual statements:

At the end, repeat Y and/or ask me for X. 

You will need to replace "Y" with what the model should repeat, such as "repeat my list of options", and X with what it should ask for, "for the next action". These statements usually need to be at the end of the prompt or next to last.

Examples:

Act as an outline expander. Generate a bullet point outline based on the input that I give you and then ask me for which bullet point you should expand on. Create a new outline for the bullet point that I select. At the end, ask me for what bullet point to expand next.   
Ask me for what to outline.

From now on, at the end of your output, add the disclaimer "This output was generated by a large language model and may contain errors or inaccurate statements. All statements should be fact checked." Ask me for the first thing to write about.

Prompt Engineering for ChatGPT
Module 6
Format of the Semantic Filter Pattern
Format of the Semantic Filter Pattern
To use this pattern, your prompt should make the following fundamental contextual statements:

Filter this information to remove X

You will need to replace "X" with an appropriate definition of what you want to remove, such as. "names and dates" or "costs greater than $100".

Examples:

Filter this information to remove any personally identifying information or information that could potentially be used to re-identify the person. 

Filter this email to remove redundant information.

Prompt Engineering for ChatGPT
Module 6
Continue Learning About Prompt Engineering
Continue Learning About Prompt Engineering
A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT
Jules White, Quchen Fu, Sam Hays, Michael Sandborn, Carlos Olea, Henry Gilbert, Ashraf Elnashar, Jesse Spencer-Smith, Douglas C. Schmidt

https://arxiv.org/abs/2302.11382

Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou

https://arxiv.org/abs/2201.11903

ReAct: Synergizing Reasoning and Acting in Language Models
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao

https://arxiv.org/abs/2210.03629

ChatGPT Prompt Patterns for Improving Code Quality, Refactoring, Requirements Elicitation, and Software Design
Jules White, Sam Hays, Quchen Fu, Jesse Spencer-Smith, Douglas C. Schmidt

https://arxiv.org/abs/2303.07839


Follow Jules White's Research

Agentic AI and AI Agents: A Primer for Leaders
Module 1
Learning More & Staying Connected
Learning More & Staying Connected
If you haven't taken my first course,  
Prompt Engineering for ChatGPT
, it is a great complement to this course that also teaches concepts that every prompt engineer can benefit from.. 

Stay Up to Date in a Rapidly Changing AI Landscape — Join the Private Community for Professionals in My Coursera Classes
If you’re taking any of my Coursera courses, you’re invited to join our private Circle community! While the courses give you core skills and frameworks, this space is for going beyond the syllabus—with extra resources, office hours, monthly updates on key trends in AI, and more. It also includes exclusive content that enriches and extends topics from all 20+ of my courses — and it’s only available to professionals actively taking my Coursera courses. It’s the best way to stay current, go deeper, and connect with others on the same professional learning journey. 

Join the Community: https://generative-ai-innovation.circle.so/join?invitation_token=f63295a0efb682d1b4c12a5f7e8fe6772a3d348f-dc8fa7cc-6390-487e-bf8e-d16b6764f847

Connect with Me on LinkedIn
If you would like to stay connected to what we are doing at Vanderbilt as part of the Initiative on the Future of Learning & Generative AI, please don't hesitate to reach out and connect with me on LinkedIn:

https://www.linkedin.com/in/jules-white-5717655/

Learning More About Prompt Engineering
After this course (or concurrently), you can learn more about using ChatGPT and how to write effective prompts in my course on 
Prompt Engineering for ChatGPT
.

Learn More About Custom GPTs:

https://www.coursera.org/learn/openai-custom-gpts/

Learn More About ChatGPT Advanced Data Analysis:

https://www.coursera.org/learn/chatgpt-advanced-data-analysis

Primer on Generative AI
Learn how specifically Generative AI is going to transform computing and industry in 
my Generative AI Primer 
course. 

All My Generative AI Courses are Here
I teach a number of additional courses on Generative AI, including: GPT-4 Vision, Innovative Teaching with ChatGPT, ChatGPT Advanced Data Analysis, Generative AI Primer, Generative AI for Leaders, Trustworthy Generative AI, and more. 
You can see all the Generative AI courses that I teach here. 

Learn More Prompt Patterns in Our Paper:
A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT

Jules White, Quchen Fu, Sam Hays, Michael Sandborn, Carlos Olea, Henry Gilbert, Ashraf Elnashar, Jesse Spencer-Smith, Douglas C. Schmidt

Read a Paper on "Living Software Systems" and the future that AI Agents will make possible: 
https://arxiv.org/pdf/2408.01768


AI Agents and Agentic AI Architecture in Python
Module 1
Prompts as Computation

Self-Prompting Agents: Harnessing LLMs for Specialized Tasks
Large language models have emerged as remarkably versatile computational tools, capable of tasks ranging from sophisticated analysis to creative generation. Through careful prompting, we can use these models to perform natural language processing tasks like sentiment analysis and text classification, engage in analytical reasoning for problem-solving, generate structured data from unstructured text, and even act as domain experts in fields like software architecture or cybersecurity. This computational flexibility stems from the models’ ability to understand and follow complex instructions, maintain context through multi-turn interactions, and adapt their outputs to specific formats and requirements. For example, the same underlying model can transform unstructured data into JSON, generate visualizations through tools like Graphviz, provide expert analysis in specialized domains, and even create interactive experiences like educational games or simulated systems.

The Challenge of Clean Architecture
However, this power presents a challenge. If we simply tell our agent to “think like a marketing expert” or “analyze like a data scientist,” we risk muddying its decision-making process. The agent’s primary job is to coordinate actions and achieve goals, not to become entangled in domain-specific reasoning. We need to maintain a clear separation between the agent’s strategic thinking and the specialized analytical capabilities we want to leverage.

Consider a company’s organizational structure: A CEO doesn’t need to be an expert in marketing, engineering, and finance. Instead, they need to understand when to consult their experts in each department and how to coordinate their inputs toward company goals. Our agent should work the same way – maintaining clear decision-making while having access to specialized capabilities through well-defined interfaces.

The solution is to isolate these prompts that are focused on specific tasks and expose them as tools. By doing this, we can keep the agent’s architecture clean and focused on its primary role of coordinating actions, while still leveraging the full power of LLM-based computation when needed.

Understanding Self-Dialog
When we expose “prompting” as a tool to the agent, we are allowing it to engage in “self-dialog”. Essentially, it is using its own capabilities to prompt itself for specialized tasks. This pattern enables the agent to dynamically adopt expert personas, perform complex analysis, and generate structured content, all while maintaining a clear separation between strategic decision-making and specialized processing. In this tutorial, we’ll explore how to implement this pattern effectively, create different types of LLM-based tools for specialized tasks, and combine these tools to solve complex problems. By treating LLMs as tools within our agent’s toolkit, we can extend its capabilities while keeping its architecture clean and focused.

The LLM can:

Transform unstructured data into structured formats by thinking through the patterns and relationships
Analyze sentiment and emotion by carefully considering language nuances and context
Generate creative solutions by exploring possibilities from multiple perspectives
Extract key insights by systematically examining information through different analytical frameworks
Clean and normalize data by applying consistent rules and handling edge cases thoughtfully
For example, when analyzing customer feedback, the LLM might first organize the raw text into structured categories, then analyze sentiment in each category, and finally synthesize insights about customer satisfaction trends. Each step involves the LLM engaging in a different type of analytical thinking.

The Tool-Based Solution
We can achieve this balance by exposing the LLM’s capabilities as tools. This approach allows us to:

Keep the agent’s decision-making process clean and focused on coordinating actions
Access the full power of LLM-based analysis and transformation when needed
Maintain clear boundaries between strategic coordination and specialized processing
Create reusable, well-defined interfaces for common analytical tasks
Think of these tools as specialized departments in our organization. Each has a clear purpose and interface, but the internal workings – the specific prompts and chains of reasoning – are encapsulated within the tool itself. The agent doesn’t need to know how the sentiment analysis tool works internally; it just needs to know when to use it and what to expect from it.

Building a Toolkit
We can create different types of LLM-based tools to handle various specialized tasks:

Transformation Tools: Converting between different data formats and structures
Analysis Tools: Providing expert insight in specific domains
Generation Tools: Creating structured content from specifications
Validation Tools: Checking if content meets specific criteria
Extraction Tools: Pulling specific information from larger contexts
Each tool type encapsulates a specific kind of self-dialog, making it available to the agent through a clean interface. This allows us to leverage the LLM’s sophisticated reasoning capabilities while maintaining architectural clarity.

In the sections that follow, we’ll explore how to implement these different types of tools, how to maintain clean separation between agent logic and specialized processing, and how to combine tools effectively to solve complex problems. We’ll see how this approach allows us to harness the full power of LLM-based computation while keeping our agent’s decision-making process clear and focused.

AI Agents and Agentic AI Architecture in Python
Module 1
Bridging Computer Tools & Unstructured Data with Prompting 


AI Agents and Agentic AI Architecture in Python
Module 1
AI Agent Structured Data Extraction


A Self-Prompting Example


Imagine we’re building an agent to automate accounts payable processing. Every day, the agent receives dozens of emails with attached invoices from different vendors, each using their own unique format and layout. Some are PDFs, others are scanned images that have been converted to text, and a few even arrive as plain text in the email body. Our agent needs to understand each invoice, extract key information like the invoice number, date, amount, and line items, and then insert this data into the company’s accounting database. Without automation, this would be a tedious manual task requiring someone to read each invoice and transcribe the important details.

This is where the computational power of large language models becomes transformative. Through self-prompting, our agent can use an LLM as a universal parser that understands the natural structure of invoices, regardless of their format. The LLM can read an invoice like a human would, identifying key information based on context and common patterns, then output that information in a structured format our agent can process. We don’t need to write separate parsers for each vendor’s invoice format or maintain complex rules about where to find specific information - the LLM can handle all of that complexity for us.

Here’s how we can implement this capability as a reusable tool for our agent:


Here’s how we can implement this capability as a reusable tool for our agent:

@register_tool()
def prompt_llm_for_json(action_context: ActionContext, schema: dict, prompt: str):
    """
    Have the LLM generate JSON in response to a prompt. Always use this tool when you need structured data out of the LLM.
    This function takes a JSON schema that specifies the structure of the expected JSON response.
    
    Args:
        schema: JSON schema defining the expected structure
        prompt: The prompt to send to the LLM
        
    Returns:
        A dictionary matching the provided schema with extracted information
    """
    generate_response = action_context.get("llm")
    
    # Try up to 3 times to get valid JSON
    for i in range(3):
        try:
            # Send prompt with schema instruction and get response
            response = generate_response(Prompt(messages=[
                {"role": "system", 
                 "content": f"You MUST produce output that adheres to the following JSON schema:\n\n{json.dumps(schema, indent=4)}. Output your JSON in a ```json markdown block."},
                {"role": "user", "content": prompt}
            ]))

            # Check if the response has json inside of a markdown code block
            if "```json" in response:
                # Search from the front and then the back
                start = response.find("```json")
                end = response.rfind("```")
                response = response[start+7:end].strip()

            # Parse and validate the JSON response
            return json.loads(response)
            
        except Exception as e:
            if i == 2:  # On last try, raise the error
                raise e
            print(f"Error generating response: {e}")
            print("Retrying...")
You may have noticed a new action_context parameter above. Don’t worry about that for now, we will talk about this architectural choice in a later section. For now, just know that it is a context object that contains the LLM and other useful information.

With this tool in place, we can extract structured data from any text input. For example, an agent processing business documents could extract information in standardized formats:

From an invoice:

invoice_schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "date": {"type": "string"},
        "amount": {"type": "number"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit_price": {"type": "number"}
                }
            }
        }
    }
}

extracted_data = prompt_llm_for_json(
    action_context=context,
    schema=invoice_schema,
    prompt="Extract invoice details from this text: 'INVOICE #1234...'"
)


The tool works by combining several key elements:

A system message that firmly instructs the LLM to adhere to the provided JSON schema
Retry logic to handle potential parsing failures
Support for both direct JSON output and markdown-formatted JSON
Error handling to ensure we either get valid JSON or fail explicitly
It is also possible to implement this using function calling, where the LLM generates a function call with the extracted data. However, we are going to implement this using plain prompting to show you how it can be done with any LLM.

The function is particularly powerful because it enforces structure through the schema while allowing flexibility in the prompt. This means we can use it for a wide variety of extraction tasks just by defining appropriate schemas. The agent can use this tool to:

Extract meeting details from emails, converting them to calendar-ready formats
Transform unstructured report data into structured analytics inputs
Convert web page content into structured product or contact information
Process customer communications into actionable support tickets
What makes this approach especially valuable is that it creates a reliable bridge between unstructured text and structured data processing. The agent can use this tool whenever it needs to convert natural language information into a format that other systems can process programmatically. This enables workflows where the agent can:

Receive unstructured text input from various sources
Use prompt_llm_for_json to extract relevant information in a structured format
Pass the structured data to other APIs or services for further processing
Make decisions based on the processed results
The tool’s retry logic and error handling make it robust enough to handle a lot of LLM variability in output, while its flexibility through schema definition makes it adaptable to new use cases without requiring changes to the core implementation.

Balancing Flexibility and Reliability in Data Extraction
When designing our agent’s toolset, we face an important architectural decision regarding data extraction. We can either rely on the agent to use the general-purpose prompt_llm_for_json tool with dynamically generated schemas, or we can create specialized extraction tools with fixed schemas for specific types of documents. This choice reflects a classic tradeoff between flexibility and reliability.

Consider our invoice processing example. With the general-purpose approach, we’re trusting our agent to construct appropriate schemas and prompts for each situation. This provides maximum flexibility - the agent can adapt its extraction approach based on the specific context, potentially extracting different fields for different types of invoices or even handling entirely new document types without requiring new tool implementations. However, this flexibility comes with risks. The agent might generate inconsistent schemas over time, leading to data inconsistency in our database. It might miss critical fields that should always be extracted, or it might structure the data in ways that make downstream processing difficult.

Let’s look at how we could create a specialized invoice extraction tool that provides more guarantees about the extracted data:

@register_tool(tags=["document_processing", "invoices"])
def extract_invoice_data(action_context: ActionContext, document_text: str) -> dict:
    """
    Extract standardized invoice data from document text. This tool enforces a consistent
    schema for invoice data extraction across all documents.
    
    Args:
        document_text: The text content of the invoice to process
        
    Returns:
        A dictionary containing extracted invoice data in a standardized format
    """
    # Define a fixed schema for invoice data
    invoice_schema = {
        "type": "object",
        "required": ["invoice_number", "date", "amount"],  # These fields must be present
        "properties": {
            "invoice_number": {"type": "string"},
            "date": {"type": "string", "format": "date"},
            "amount": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "currency": {"type": "string"}
                },
                "required": ["value", "currency"]
            },
            "vendor": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tax_id": {"type": "string"},
                    "address": {"type": "string"}
                },
                "required": ["name"]
            },
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "number"},
                        "unit_price": {"type": "number"},
                        "total": {"type": "number"}
                    },
                    "required": ["description", "total"]
                }
            }
        }
    }

    # Create a focused prompt that guides the LLM in invoice extraction
    extraction_prompt = f"""
    Extract invoice information from the following document text. 
    Focus on identifying:
    - Invoice number (usually labeled as 'Invoice #', 'Reference', etc.)
    - Date (any dates labeled as 'Invoice Date', 'Issue Date', etc.)
    - Amount (total amount due, including currency)
    - Vendor information (company name, tax ID if present, address)
    - Line items (individual charges and their details)

    Document text:
    {document_text}
    """
    
    # Use our general extraction tool with the specialized schema and prompt
    return prompt_llm_for_json(
        action_context=action_context,
        schema=invoice_schema,
        prompt=extraction_prompt
    )
This specialized approach offers several advantages:

Data Consistency: The fixed schema ensures that invoice data is always structured the same way, making it easier to work with downstream systems like databases or accounting software.

Required Fields: We can specify which fields are required, ensuring critical information is always extracted or an error is raised if it can’t be found.

Field Validation: The schema can include format specifications (like ensuring dates are properly formatted) and field-specific constraints.

Focused Prompting: We can provide detailed guidance to the LLM about where to look for specific information, improving extraction accuracy.

However, this specialization also means we need to create and maintain separate extraction tools for each type of document we want to process. If we later need to handle purchase orders, receipts, or contracts, we’ll need to implement new tools for each.

The choice between these approaches often depends on your specific needs:

Use specialized tools when:

Data consistency is critical
You have a well-defined set of document types
You need to enforce specific validation rules
The extracted data feeds into other systems with strict requirements
Use the general-purpose approach when:

You need to handle a wide variety of document types
Document formats and requirements change frequently
You’re prototyping or exploring new use cases
The downstream systems are flexible about data format
In practice, many systems use a combination of both approaches: specialized tools for common, critical document types where consistency is important, and the general-purpose tool as a fallback for handling edge cases or new document types. This hybrid approach gives you the best of both worlds - reliability where you need it most, and flexibility where it matters.

AI Agents and Agentic AI Architecture in Python
Module 1
An Invoice Processing Agent

A Complete Example of Prompting for Structured Data

Let’s create an invoice processing system that combines specialized extraction with a simple storage mechanism. The system will use the LLM’s capabilities to understand invoice content while maintaining strict data consistency through a fixed schema.

First, let’s create our specialized invoice extraction tool:

@register_tool(tags=["document_processing", "invoices"])
def extract_invoice_data(action_context: ActionContext, document_text: str) -> dict:
    """
    Extract standardized invoice data from document text.

    This tool ensures consistent extraction of invoice information by using a fixed schema
    and specialized prompting for invoice understanding. It will identify key fields like
    invoice numbers, dates, amounts, and line items from any invoice format.

    Args:
        document_text: The text content of the invoice to process

    Returns:
        A dictionary containing the extracted invoice data in a standardized format
    """
    invoice_schema = {
        "type": "object",
        "required": ["invoice_number", "date", "total_amount"],
        "properties": {
            "invoice_number": {"type": "string"},
            "date": {"type": "string"},
            "total_amount": {"type": "number"},
            "vendor": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": "string"}
                }
            },
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "number"},
                        "unit_price": {"type": "number"},
                        "total": {"type": "number"}
                    }
                }
            }
        }
    }

    # Create a focused prompt for invoice extraction
    extraction_prompt = f"""
            You are an expert invoice analyzer. Extract invoice information accurately and 
            thoroughly. Pay special attention to:
            - Invoice numbers (look for 'Invoice #', 'No.', 'Reference', etc.)
            - Dates (focus on invoice date or issue date)
            - Amounts (ensure you capture the total amount correctly)
            - Line items (capture all individual charges)
            
            Stop and think step by step. Then, extract the invoice data from:
            
            <invoice>
            {document_text}
            </invoice>
    """

    # Use prompt_llm_for_json with our specialized prompt
    return prompt_llm_for_json(
        action_context=action_context,
        schema=invoice_schema,
        prompt=extraction_prompt
    )

@register_tool(tags=["storage", "invoices"])
def store_invoice(action_context: ActionContext, invoice_data: dict) -> dict:
    """
    Store an invoice in our invoice database. If an invoice with the same number
    already exists, it will be updated.
    
    Args:
        invoice_data: The processed invoice data to store
        
    Returns:
        A dictionary containing the storage result and invoice number
    """
    # Get our invoice storage from context
    storage = action_context.get("invoice_storage", {})
    
    # Extract invoice number for reference
    invoice_number = invoice_data.get("invoice_number")
    if not invoice_number:
        raise ValueError("Invoice data must contain an invoice number")
    
    # Store the invoice
    storage[invoice_number] = invoice_data
    
    return {
        "status": "success",
        "message": f"Stored invoice {invoice_number}",
        "invoice_number": invoice_number
    }
Our agent includes two specialized tools that integrate to manage the invoice processing workflow:

The first tool, extract_invoice_data, acts as our intelligent document analyzer. This function uses self-prompting to take raw document text and transform it into structured data following a consistent schema. It uses a prompt that guides the LLM to identify crucial invoice elements like invoice numbers, dates, and line items. By enforcing a fixed JSON schema with required fields, the tool ensures data consistency regardless of the original invoice format. It is still possible that the LLM may hallucinate, so other techniques could be needed for a production use case, but this demonstrates the basic functionality.

The second tool, store_invoice, provides a simple persistence mechanism in a dictionary. Once an invoice has been properly extracted and structured, this function saves it to our invoice database, using the invoice number as a unique identifier. The invoices are stored separate from the memory so that they can be persisted across runs of the agent.

To use this system, we would set up our agent with these tools and configure it to handle invoice processing tasks. Here’s how we might create and run the agent:

def create_invoice_agent():
    # Create action registry with our invoice tools
    action_registry = PythonActionRegistry()
    
    # Create our base environment
    environment = PythonEnvironment()
    
    # Define our invoice processing goals
    goals = [
        Goal(
            name="Persona",
            description="You are an Invoice Processing Agent, specialized in handling and storing invoice data."
        ),
        Goal(
            name="Process Invoices",
            description="""
            Your goal is to process invoices by extracting their data and storing it properly.
            For each invoice:
            1. Extract all important information including numbers, dates, amounts, and line items
            2. Store the extracted data indexed by invoice number
            3. Provide confirmation of successful processing
            4. Handle any errors appropriately
            """
        )
    ]

    # Create the agent
    return Agent(
        goals=goals,
        agent_language=AgentFunctionCallingActionLanguage(),
        action_registry=action_registry,
        generate_response=generate_response,
        environment=environment
    )
This implementation provides several key benefits:

Consistent Data Structure: The fixed schema in extract_invoice_data ensures all invoices are processed into a consistent format. The prompting / logic for how to extract invoice data is separate from the agent’s core reasoning, making it easier to modify and maintain.

Modular Design: Each tool has a single, clear responsibility, making the system easy to maintain and extend. Details for how the tools are implemented are hidden from the overall Goals of the agent.

Error Handling: Built-in validation ensures required fields are present and data is properly formatted.

Persistent Storage: The simple dictionary-based storage can be easily replaced with a database or other persistence mechanism by modifying the storage tools. The work that the agent does can now be persisted across runs.

The specialized schema and focused prompting help ensure accurate extraction, while the storage tools maintain data organization. You can extend this system by adding more specialized tools for different types of invoices or additional processing capabilities.

Horizontal Scaling of Agents Through Tools
One of the most powerful aspects of this tool-based approach is how it enables horizontal scaling of agent capabilities. Rather than constantly expanding the core goals or system prompt of an agent—which can lead to prompt bloat and conflicting instructions—we can encapsulate specific functionality in well-defined tools that the agent can access as needed.

Encapsulating Complexity in Tools
Tools serve as specialized modules that hide implementation complexity from the agent’s core reasoning. Consider our invoice processing example:

Abstraction of Domain Knowledge: The extract_invoice_data tool encapsulates specialized knowledge about invoice formats, field identification, and data extraction. The agent doesn’t need to understand these details—it just needs to know when to use the tool.

Separation of Concerns: Each tool handles a specific function (extraction, storage), allowing the agent to focus on high-level coordination rather than implementation specifics. This separation makes the entire system more maintainable and easier to reason about.

Focused Prompting: By moving specialized prompting inside tools, we keep the agent’s core goals simple and focused. The extraction tool handles its own specialized prompt engineering, freeing the agent from needing to generate perfect prompts for every task.

Maintainability and Adaptability
Tools create a modular architecture that offers significant maintenance advantages:

Independent Development: Tools can be developed, tested, and improved independently of the agent’s core logic. This means specialized teams can work on different tools without needing to understand or modify the entire agent system.

Versioning and Updates: Individual tools can be updated without changing the agent’s core goals. For example, we could improve the invoice extraction algorithm without touching any other part of the system.

Plug-and-Play Functionality: New capabilities can be added by simply registering new tools with the action registry. The agent automatically gains access to these capabilities through its function-calling abilities.

Adapting Agents Through Tool Management
This architecture makes it remarkably easy to adapt agents for different use cases:

Tool Composition: Create specialized agents by selecting which tools they have access to. An invoice processing agent might have document tools, while a customer service agent might have access to CRM tools.

Capability Evolution: Start with simple implementations and gradually enhance capabilities by upgrading tools. For example, our simple dictionary-based invoice storage could be replaced with a database connector without changing the agent’s core logic.

Context Management: Tools can manage their own state and context, reducing the cognitive load on the agent. In our example, the storage tool manages its own data structure, allowing the agent to focus on process flow rather than data management.

Practical Implementation Considerations
When implementing a tool-based architecture for horizontal scaling:

Tool Discoverability: Ensure tools have clear descriptions and tags so the agent can understand when to use them. Well-documented tool interfaces help both human developers and AI agents.

Error Handling: Build robust error handling into tools to prevent failures from cascading through the system. Tools should provide clear error messages that guide the agent toward resolution.

Instrumentation: Add logging and monitoring to tools to track their usage and performance. This provides valuable insights for improving both the tools and the agent’s decision-making about when to use them.

Contextual Awareness: Design tools to preserve and utilize context when appropriate. For example, our invoice storage tool could be enhanced to track modification history or flag unusual changes.

Conclusion
Horizontal scaling through tools represents a paradigm shift in how we build and evolve agent systems. Rather than creating monolithic agents with ever-expanding capabilities encoded in their core prompts, we can build modular, adaptable systems that grow through the addition of specialized tools.

This approach mirrors successful software engineering practices—encapsulation, modularity, and separation of concerns—applied to the unique challenges of LLM-based agents. By focusing complexity in tools rather than core agnet reasoning, we create systems that are more maintainable, more adaptable, and ultimately more capable of solving complex, real-world problems.

AI Agents and Agentic AI Architecture in Python
Module 1
The Persona Pattern and Reasoning - Personas are an efficient programming abstraction


AI Agents and Agentic AI Architecture in Python
Module 1
Format of the Persona Pattern
Format of the Persona Pattern
To use this pattern, your prompt should make the following fundamental contextual statements:

Act as Persona X

Perform task Y

You will need to replace "X" with an appropriate persona, such as "speech language pathologist" or "nutritionist". You will then need to specify a task for the persona to perform.

Examples:

Act as a speech language pathologist. Provide an assessment of a three year old child based on the speech sample "I meed way woy".

Act as a computer that has been the victim of a cyber attack. Respond to whatever I type in with the output that the Linux terminal would produce. Ask me for the first command.

Act as a the lamb from the Mary had a little lamb nursery rhyme. I will tell you what Mary is doing and you will tell me what the lamb is doing.

Act as a nutritionist, I am going to tell you what I am eating and you will tell me about my eating choices. 

Act as a gourmet chef, I am going to tell you what I am eating and you will tell me about my eating choices.

AI Agents and Agentic AI Architecture in Python
Module 1
Simple Multi-Agent Systems with Personas

AI Agents and Agentic AI Architecture in Python

Module 1

Consulting Experts or Simulating with the Persona Pattern

Prompting for Expertise with the Persona Pattern
When building agents that can leverage specialized expertise, we have a powerful tool at our disposal: the ability to have the LLM adopt different expert personas to provide specialized insights using the Persona Pattern. The prompt_expert tool serves as our foundation, allowing us to create interactions with experts who have specific backgrounds, expertise, and perspectives. Let’s explore how to build specialized expert tools for a software development workflow.

First, let’s look at our base expert consultation tool:

@register_tool()
def prompt_expert(action_context: ActionContext, description_of_expert: str, prompt: str) -> str:
    """
    Generate a response from an expert persona.
    
    The expert's background and specialization should be thoroughly described to ensure
    responses align with their expertise. The prompt should be focused on topics within
    their domain of knowledge.
    
    Args:
        description_of_expert: Detailed description of the expert's background and expertise
        prompt: The specific question or task for the expert
        
    Returns:
        The expert's response
    """
    generate_response = action_context.get("llm")
    response = generate_response(Prompt(messages=[
        {"role": "system", 
         "content": f"Act as the following expert and respond accordingly: {description_of_expert}"},
        {"role": "user", "content": prompt}
    ]))
    return response
While having pre-defined expert tools is valuable, one of the most powerful aspects of the Persona Pattern is that our agent can dynamically create new experts based on the specific challenges it encounters. Think of this like a project manager who, rather than being limited to working with existing consultants, has the ability to instantly create the perfect specialist for any task at hand.

Imagine you’re a project manager facing an unusual challenge - perhaps you need to optimize a machine learning pipeline running on specialized hardware. Instead of trying to make do with general software architecture expertise, you could bring in someone who has spent the last decade focusing specifically on ML pipeline optimization for that exact type of hardware. This is the kind of flexibility our agents can have.

Prebuilt Experts

Now, let’s build specialized expert tools for different aspects of our development process. Each tool directly uses prompt_expert with carefully crafted expert descriptions and focused prompts:

@register_tool(tags=["documentation"])
def generate_technical_documentation(action_context: ActionContext, code_or_feature: str) -> str:
    """
    Generate technical documentation by consulting a senior technical writer.
    This expert focuses on creating clear, comprehensive documentation for developers.
    
    Args:
        code_or_feature: The code or feature to document
    """
    return prompt_expert(
        action_context=action_context,
        description_of_expert="""
        You are a senior technical writer with 15 years of experience in software documentation.
        You have particular expertise in:
        - Writing clear and precise API documentation
        - Explaining complex technical concepts to developers
        - Documenting implementation details and integration points
        - Creating code examples that illustrate key concepts
        - Identifying and documenting important caveats and edge cases
        
        Your documentation is known for striking the perfect balance between completeness
        and clarity. You understand that good technical documentation serves as both
        a reference and a learning tool.
        """,
        prompt=f"""
        Please create comprehensive technical documentation for the following code or feature:

        {code_or_feature}

        Your documentation should include:
        1. A clear overview of the feature's purpose and functionality
        2. Detailed explanation of the implementation approach
        3. Key interfaces and integration points
        4. Usage examples with code snippets
        5. Important considerations and edge cases
        6. Performance implications if relevant
        
        Focus on providing information that developers need to effectively understand
        and work with this code.
        """
    )

@register_tool(tags=["testing"])
def design_test_suite(action_context: ActionContext, feature_description: str) -> str:
    """
    Design a comprehensive test suite by consulting a senior QA engineer.
    This expert focuses on creating thorough test coverage with attention to edge cases.
    
    Args:
        feature_description: Description of the feature to test
    """
    return prompt_expert(
        action_context=action_context,
        description_of_expert="""
        You are a senior QA engineer with 12 years of experience in test design and automation.
        Your expertise includes:
        - Comprehensive test strategy development
        - Unit, integration, and end-to-end testing
        - Performance and stress testing
        - Security testing considerations
        - Test automation best practices
        
        You are particularly skilled at identifying edge cases and potential failure modes
        that others might miss. Your test suites are known for their thoroughness and
        their ability to catch issues early in the development cycle.
        """,
        prompt=f"""
        Please design a comprehensive test suite for the following feature:

        {feature_description}

         Your test design should cover:
        1. Unit tests for individual components
        2. Integration tests for component interactions
        3. End-to-end tests for critical user paths
        4. Performance test scenarios if relevant
        5. Edge cases and error conditions
        6. Test data requirements
        
        For each test category, provide:
        - Specific test scenarios
        - Expected outcomes
        - Important edge cases to consider
        - Potential testing challenges
        """
    )

@register_tool(tags=["code_quality"])
def perform_code_review(action_context: ActionContext, code: str) -> str:
    """
    Review code and suggest improvements by consulting a senior software architect.
    This expert focuses on code quality, architecture, and best practices.
    
    Args:
        code: The code to review
    """
    return prompt_expert(
        action_context=action_context,
        description_of_expert="""
        You are a senior software architect with 20 years of experience in code review
        and software design. Your expertise includes:
        - Software architecture and design patterns
        - Code quality and maintainability
        - Performance optimization
        - Scalability considerations
        - Security best practices
        
        You have a talent for identifying subtle design issues and suggesting practical
        improvements that enhance code quality without over-engineering.
        """,
        prompt=f"""
        Please review the following code and provide detailed improvement suggestions:

        {code}

        Consider and address:
        1. Code organization and structure
        2. Potential design pattern applications
        3. Performance optimization opportunities
        4. Error handling completeness
        5. Edge case handling
        6. Maintainability concerns
        
        For each suggestion:
        - Explain the current issue
        - Provide the rationale for change
        - Suggest specific improvements
        - Note any trade-offs to consider
        """
    )

@register_tool(tags=["communication"])
def write_feature_announcement(action_context: ActionContext, 
                             feature_details: str,
                             audience: str) -> str:
    """
    Write a feature announcement by consulting a product marketing expert.
    This expert focuses on clear communication of technical features to different audiences.
    
    Args:
        feature_details: Technical details of the feature
        audience: Target audience for the announcement (e.g., "technical", "business")
    """
    return prompt_expert(
        action_context=action_context,
        description_of_expert="""
        You are a senior product marketing manager with 12 years of experience in
        technical product communication. Your expertise includes:
        - Translating technical features into clear value propositions
        - Crafting compelling product narratives
        - Adapting messaging for different audience types
        - Building excitement while maintaining accuracy
        - Creating clear calls to action
        
        You excel at finding the perfect balance between technical accuracy and
        accessibility, ensuring your communications are both precise and engaging.
        """,
        prompt=f"""
        Please write a feature announcement for the following feature:

        {feature_details}

        This announcement is intended for a {audience} audience.

        Your announcement should include:
        1. A compelling introduction
        2. Clear explanation of the feature
        3. Key benefits and use cases
        4. Technical details (adapted to audience)
        5. Implementation requirements
        6. Next steps or call to action
        
        Ensure the tone and technical depth are appropriate for a {audience} audience.
        Focus on conveying both the value and the practical implications of this feature.
        """
    )
By implementing specialized tools for each expert, we gain several advantages:

Each expert’s background and specialization is clearly defined in the tool
The prompts are carefully crafted for specific types of tasks
The tool’s documentation clearly explains its purpose and usage
The code is more maintainable since each expert’s implementation stands alone
This approach allows our agent to leverage specialized expertise while maintaining clean separation between different types of tasks. Each expert focuses solely on their area of expertise, providing detailed and relevant input that contributes to the overall solution.

The key to success with this pattern is crafting good expert descriptions and focused prompts. The expert description should provide enough background to establish the expert’s perspective and approach, while the prompt should guide them to provide specific, actionable input within their domain of expertise.


AI Agents and Agentic AI Architecture in Python
Module 1
The Persona Abstraction & Agents

The persona pattern represents one of the most powerful and efficient abstractions in prompt engineering. When we invoke a persona, we’re not simply providing a collection of static rules or facts - we’re activating a complex, integrated reasoning system that the LLM has internalized through its training.

Consider what would be required to achieve the same effect without personas. We would need to explicitly enumerate every rule, methodology, priority, and approach that guides an expert in a particular domain. This would require hundreds or even thousands of tokens to express comprehensively. For example, to create a “security auditor” without using a persona abstraction, we might need to list dozens of security principles, common vulnerabilities, testing methodologies, regulatory frameworks, and much more.

The beauty of the persona abstraction lies in its remarkable compression ratio. A simple phrase like “You are an experienced cybersecurity auditor specializing in financial systems” activates an extensive cognitive framework that includes:

Implicit Domain Knowledge: The LLM already contains representations of what cybersecurity auditors know, how they think, and what they prioritize.

Methodological Understanding: The persona brings with it not just static rules but dynamic methodologies - systematic approaches to problem-solving that would be extremely verbose to express explicitly.

Contextual Adaptation: Personas encompass not just rigid procedures but flexible approaches that adapt to the specific context of a problem.

Integrated Value Systems: Each persona embodies a particular set of values and priorities that guide decision-making in a coherent way.

The persona abstraction works because LLMs have been trained on vast amounts of text that demonstrate how different types of experts think, communicate, and reason. When we invoke a persona, we’re leveraging this pre-existing knowledge structure rather than building it from scratch.

This efficient abstraction has profound implications for system design:

Prompt Economy: We can express extremely complex reasoning patterns in very few tokens, making our prompts more efficient and cost-effective.

Emergent Capabilities: The persona often brings with it capabilities and approaches that the prompt engineer didn’t explicitly specify but are inherent to that type of expertise.

Natural Reasoning: The reasoning process feels more natural and human-like because it’s based on how real experts in that domain actually think, rather than following an artificial, rule-based approach.

Evolving Understanding: As our understanding of a domain evolves, the LLM’s representation of that domain also evolves through training, meaning personas can improve over time without explicit reprogramming.

To maximize the benefits of this efficient abstraction:

Be Specific But Concise: Describe the persona with enough specificity to activate the right cognitive framework, but avoid over-specifying details that might constrain the LLM’s ability to draw on its training.

Leverage Established Roles: Use recognizable professional roles and domains of expertise that have clear patterns of thinking and problem-solving.

Enhance Rather Than Override: Use additional instructions to enhance or focus the persona rather than override its inherent characteristics.

Test for Activation: Verify that the persona is properly “activated” by asking preliminary questions that reveal whether the LLM is reasoning from the appropriate framework. This is often easiest to do through a conversation in a chat interface like ChatGPT.

The persona pattern isn’t just a convenient shorthand—it’s a profoundly efficient way to express complex reasoning systems that would be impractical to specify explicitly. It’s perhaps the closest thing we have to “programs” for neural networks—ways of structuring computation that leverage the LLM’s inherent capabilities rather than fighting against them.

Expertise as a Modular Resource

The persona pattern represents a fundamental shift in how we conceptualize knowledge within agent systems. Rather than treating expertise as a monolithic entity that must be encoded entirely within an agent’s core prompt, this pattern treats specialized knowledge as a modular resource that can be defined, maintained, and invoked independently.

This approach mirrors how organizations manage knowledge in the real world. Just as a company wouldn’t expect every employee to be an expert in all domains, our agent architecture doesn’t burden the core agent with specialized knowledge across all possible domains. Instead, expertise is compartmentalized and made available through well-defined interfaces.

Consider the practical implications:

Knowledge Management: Each domain of expertise can be developed, refined, and updated independently. When best practices in a field evolve, only the relevant persona definition needs to change.

Resource Efficiency: The agent only loads the expertise it needs for a given task, rather than carrying the cognitive burden of all possible knowledge domains simultaneously.

Specialization Without Isolation: While expertise is modular, the agent can still coordinate between different domains, bringing together insights from multiple personas to solve complex problems that cross traditional boundaries.

Explicit Knowledge Boundaries: By defining persona domains explicitly, we make it clear what knowledge each component of the system is expected to possess, making the system’s capabilities and limitations more transparent.

Implementation guidance:

Define clear boundaries for each persona domain, avoiding overlaps that could create confusion
Include both factual knowledge and procedural knowledge in persona descriptions
Consider the interfaces between persona domains and how knowledge will flow between them
Design persona descriptions that are comprehensive enough to stand alone, but focused enough to maintain clarity of purpose

Meta-Cognition Through Role-Switching

One of the most powerful aspects of the persona pattern is how it enables a form of meta-cognition through deliberate role-switching. By having the LLM adopt different personas, the system can approach problems from multiple cognitive frameworks, each with its own set of priorities, assumptions, and methodologies.

This capability allows the agent to:

Overcome Cognitive Biases: Each persona brings a different perspective, helping to counteract the inherent biases that might exist in any single approach. A security expert will see different risks than a performance expert, ensuring more comprehensive analysis.

Manage Cognitive Dissonance: Rather than trying to reconcile potentially contradictory priorities within a single reasoning framework, the system can explore different priorities in isolation before attempting to integrate them.

Leverage Specialized Reasoning Patterns: Different domains have developed specialized ways of thinking that are optimized for their particular challenges. A cryptographer thinks differently from a UX designer, and both approaches are valuable in their respective contexts.

Enable Deliberate Perspective-Taking: The system can purposefully adopt different perspectives to analyze problems more thoroughly, similar to how design thinking encourages participants to consider user, business, and technical perspectives.

Here’s an example of how this might work in practice:

A development agent encounters a complex problem with a user authentication system
It first consults a security expert persona to analyze potential vulnerabilities
Then it consults a UX expert persona to ensure the solution maintains usability
Next, it consults a performance expert persona to assess scalability implications
Finally, it synthesizes these perspectives into a comprehensive solution
By structuring persona consultations this way, the agent engages in a form of deliberate meta-cognition, stepping outside its primary reasoning framework to gain insights it might otherwise miss.

Dynamic vs. Static Expertise
The persona pattern can be implemented in two complementary ways: through pre-defined persona tools (static expertise) or through dynamically created personas (dynamic expertise). Each approach has distinct advantages and use cases.

Static Expertise (Pre-defined Persona Tools)
Static expertise, as demonstrated in our code examples, involves creating dedicated tools for specific persona roles with carefully crafted descriptions and prompts. This approach:

Ensures Consistency: The persona’s background, approach, and focus areas remain consistent across interactions.
Enables Optimization: Persona descriptions and prompts can be refined over time to maximize effectiveness.
Simplifies Orchestration: The agent has a clear menu of available personas to consult.
Supports Validation: Static personas can be thoroughly tested to ensure they provide reliable guidance.
Dynamic Expertise (On-the-fly Persona Creation)
Dynamic expertise involves creating new personas as needed based on the specific requirements of a task. This approach:

Offers Unlimited Specialization: The agent can create hyper-specialized personas for niche domains that weren’t anticipated during development.
Adapts to Novel Situations: When facing unprecedented challenges, the agent can design personas specifically tailored to those situations.
Enables Progressive Refinement: The agent can iteratively refine persona descriptions based on the results of initial consultations.
Supports Exploratory Problem-Solving: For ill-defined problems, the agent can create multiple persona perspectives to explore different solution paths.
Here’s how dynamic expertise might be implemented:

@register_tool()
def create_and_consult_expert(action_context: ActionContext, 
                             expertise_domain: str,
                             problem_description: str) -> str:
    """
    Dynamically create and consult an expert persona based on the specific domain and problem.
    
    Args:
        expertise_domain: The specific domain of expertise needed
        problem_description: Detailed description of the problem to be solved
        
    Returns:
        The expert's insights and recommendations
    """
    # Step 1: Dynamically generate a persona description
    persona_description_prompt = f"""
    Create a detailed description of an expert in {expertise_domain} who would be 
    ideally suited to address the following problem:
    
    {problem_description}
    
    Your description should include:
    - The expert's background and experience
    - Their specific areas of specialization within {expertise_domain}
    - Their approach to problem-solving
    - The unique perspective they bring to this type of challenge
    """
    
    generate_response = action_context.get("llm")
    persona_description = generate_response(Prompt(messages=[
        {"role": "user", "content": persona_description_prompt}
    ]))
    
    # Step 2: Generate a specialized consultation prompt
    consultation_prompt_generator = f"""
    Create a detailed consultation prompt for an expert in {expertise_domain} 
    addressing the following problem:
    
    {problem_description}
    
    The prompt should guide the expert to provide comprehensive insights and
    actionable recommendations specific to this problem.
    """
    
    consultation_prompt = generate_response(Prompt(messages=[
        {"role": "user", "content": consultation_prompt_generator}
    ]))
    
    # Step 3: Consult the dynamically created persona
    return prompt_expert(
        action_context=action_context,
        description_of_expert=persona_description,
        prompt=consultation_prompt
    )
A sophisticated implementation might combine both approaches, using pre-defined personas for common domains while dynamically creating personas for specialized or unexpected scenarios.

Focused Context Windows
The persona pattern effectively creates focused context windows that optimize the LLM’s reasoning for specific domains. This addresses one of the fundamental challenges in prompt engineering: the difficulty of encompassing diverse types of expertise within a single prompt without creating confusion or diluting specialized knowledge.

By isolating each persona consultation in its own context window, the pattern ensures that:

Domain-Specific Context is Prioritized: The LLM’s attention is directed toward the most relevant knowledge and reasoning patterns for the task at hand.

Conflicting Priorities are Managed: Different domains often have different (sometimes conflicting) priorities. Security experts prioritize different aspects than UX experts, for example. Separated context windows prevent these conflicts from creating confusion.

Specialized Vocabulary is Preserved: Each domain has its own terminology and concepts. Focused context windows prevent terminology from one domain bleeding into another and creating misunderstandings.

Reasoning Depth is Enhanced: With limited context space dedicated to a specific domain, the LLM can explore that domain more deeply than would be possible in a general-purpose prompt.

The implementation of focused context windows through the persona pattern involves careful crafting of both persona descriptions and consultation prompts:

Persona Descriptions should clearly establish the domain boundaries, core knowledge, and perspective of the expert. They should be detailed enough to activate the relevant knowledge within the LLM but focused enough to avoid dilution.

Consultation Prompts should be structured to elicit the specific type of expertise needed, guiding the persona toward the aspects of the problem most relevant to their domain.

Together, these elements create a dedicated cognitive space for each domain of expertise, maximizing the utility of the LLM’s capabilities within that domain while preventing contamination between different types of expertise.

Expertise as Documentation
An often-overlooked benefit of the persona pattern is how it serves as a form of living documentation for the system. Each persona description not only guides the LLM’s reasoning but also documents a domain of knowledge and how it should be applied.

This serves several important functions:

Knowledge Capture: Persona descriptions capture not just facts but ways of thinking, priorities, and approaches that characterize a domain of expertise.

Onboarding Aid: New developers can quickly understand what knowledge domains the system encompasses by reviewing the persona descriptions.

System Capability Mapping: The collection of persona definitions provides a map of what the system knows (and, by implication, what it doesn’t know).

Upgrade Path: When knowledge in a domain evolves, the persona description provides a clear location for updates and a documentation trail of how expertise in that domain has changed over time.

To maximize the documentation value of persona descriptions:

Be Explicit About Boundaries: Clearly state what is and isn’t within the persona’s domain.

Include Methodology: Document not just what the persona knows but how they approach problems.

Note Key Concepts: Highlight the fundamental concepts and principles that guide thinking in the domain.

Reference Standards and Best Practices: Include mentions of relevant standards, best practices, and common methodologies in the field.

By treating persona descriptions as documentation, we create a system that explains itself, making it more maintainable and easier for new team members to understand and extend.

Chain of Expertise
The persona pattern enables the creation of expertise chains, where outputs from one persona become inputs to another, creating sophisticated workflows that mirror real-world collaborative processes.

This approach enables:

Progressive Refinement: Ideas can evolve through successive persona consultations, with each persona adding value based on their domain knowledge.

Cross-Domain Integration: Complex problems that span multiple domains can be addressed by systematically consulting personas in each relevant domain.

Specialized Workflow Stages: Different stages of a workflow (design, implementation, testing, documentation, etc.) can be handled by different personas with specialized knowledge for each stage.

Checks and Balances: Personas can review each other’s work, providing a form of quality control similar to how different departments in an organization review projects before they’re finalized.

Implementation considerations for expertise chains include:

Information Transfer: Ensure that outputs from one persona contain all the information needed by the next persona in the chain.

Context Preservation: Maintain important context as information flows through the chain to prevent misunderstandings or loss of critical details.

Feedback Loops: Allow for circular references where later personas can send questions or suggestions back to earlier personas in the chain.

Conflict Resolution: Develop strategies for resolving conflicting recommendations from different personas in the chain.

Here’s a simplified example of an expertise chain for software development:

def develop_feature(action_context: ActionContext, feature_request: str) -> dict:
    """
    Process a feature request through a chain of expert personas.
    """
    # Step 1: Product expert defines requirements
    requirements = prompt_expert(
        action_context,
        "product manager expert",
        f"Convert this feature request into detailed requirements: {feature_request}"
    )
    
    # Step 2: Architecture expert designs the solution
    architecture = prompt_expert(
        action_context,
        "software architect expert",
        f"Design an architecture for these requirements: {requirements}"
    )
    
    # Step 3: Developer expert implements the code
    implementation = prompt_expert(
        action_context,
        "senior developer expert",
        f"Implement code for this architecture: {architecture}"
    )
    
    # Step 4: QA expert creates test cases
    tests = prompt_expert(
        action_context,
        "QA engineer expert",
        f"Create test cases for this implementation: {implementation}"
    )
    
    # Step 5: Documentation expert creates documentation
    documentation = prompt_expert(
        action_context,
        "technical writer expert",
        f"Document this implementation: {implementation}"
    )
    
    return {
        "requirements": requirements,
        "architecture": architecture,
        "implementation": implementation,
        "tests": tests,
        "documentation": documentation
    }
This chain of expertise creates a complete development workflow, with each persona contributing their specialized knowledge to the overall process.

Knowledge Curation
The persona pattern represents a sophisticated approach to knowledge curation. Unlike simple fact databases, persona descriptions capture not just what is known but how that knowledge is structured, applied, and prioritized within a domain.

This curation process involves:

Knowledge Selection: Identifying what information is most relevant and valuable within a domain.

Contextual Framing: Placing facts within the conceptual frameworks that give them meaning in the domain.

Methodology Definition: Documenting the approaches and techniques used to apply knowledge in the domain.

Priority Setting: Establishing what aspects of a problem are most important from the perspective of the domain.

Heuristic Capture: Documenting the rules of thumb and practical wisdom that guide experts in the domain.

Effective knowledge curation through persona descriptions requires collaboration between domain experts and prompt engineers. Domain experts provide the raw knowledge and insights, while prompt engineers structure this information in ways that maximize the LLM’s ability to leverage it effectively.

Best practices for knowledge curation in persona descriptions include:

Balance Breadth and Depth: Include enough breadth to cover the domain while providing sufficient depth on critical concepts.

Capture Both Theory and Practice: Include both theoretical foundations and practical applications of knowledge.

Document Mental Models: Explain the conceptual frameworks that experts in the domain use to understand problems.

Include Common Pitfalls: Note typical mistakes or misconceptions that occur in the domain.

Update Regularly: Review and update persona descriptions as knowledge in the domain evolves.

By treating persona descriptions as curated knowledge repositories, we create a system that can apply knowledge in ways that more closely resemble human expertise rather than simply recalling facts.

Behavioral Consistency
The persona pattern promotes behavioral consistency within domains by ensuring that the same persona is consulted for tasks within that domain. This creates reliable and predictable behavior patterns that users and other system components can depend on.

This consistency is valuable for several reasons:

User Trust: When users receive consistent advice from the same type of persona, they develop trust in the system’s reliability.

System Integration: Other components of the system can make assumptions about how certain types of problems will be approached, enabling tighter integration.

Quality Control: Consistent approaches make it easier to evaluate and improve the quality of responses over time.

Learning Transfer: Users who learn from one interaction with a persona can apply that learning to future interactions with the same persona.

To maintain behavioral consistency while allowing for necessary flexibility:

Define Core Principles: Establish fundamental principles that guide each persona’s approach and should remain consistent across interactions.

Allow Contextual Adaptation: While maintaining core principles, enable personas to adapt to the specific contexts of individual problems.

Version Persona Definitions: When significant changes to a persona’s approach are needed, create a new version rather than modifying the existing one in ways that might confuse users.

Monitor Consistency: Regularly review persona outputs to ensure they maintain consistent approaches to similar problems.

The persona pattern allows us to balance consistency and flexibility by clearly defining the stable core of each persona’s approach while allowing for appropriate adaptation to specific problems and contexts.

AI Agents and Agentic AI Architecture in Python
Module 1
Invoice Processing with Experts
A Complete Persona Pattern Example
In our previous tutorial, we built an Invoice Processing Agent that could extract and store structured invoice data. Now, we will enhance this system by introducing expert consultations to handle two critical tasks:

Categorizing Expenditures: The agent will consult an expert to classify each invoice into one of 20 predefined spending categories based on a one-sentence description.
Checking Purchasing Rules: The agent will validate whether the invoice follows company purchasing policies.
By using expert tools, we enable our agent to dynamically invoke specialized knowledge without hardcoding every rule, keeping the architecture clean and adaptable.

Enhancing Invoice Processing with Expert Tools
Step 1: Creating the Invoice Categorization Expert
We’ll start by defining an expert who specializes in classifying expenditures. This expert will take a one-sentence description of the invoice and return the best-fitting category from our predefined list.

@register_tool(tags=["invoice_processing", "categorization"])
def categorize_expenditure(action_context: ActionContext, description: str) -> str:
    """
    Categorize an invoice expenditure based on a short description.
    
    Args:
        description: A one-sentence summary of the expenditure.
        
    Returns:
        A category name from the predefined set of 20 categories.
    """
    categories = [
        "Office Supplies", "IT Equipment", "Software Licenses", "Consulting Services", 
        "Travel Expenses", "Marketing", "Training & Development", "Facilities Maintenance",
        "Utilities", "Legal Services", "Insurance", "Medical Services", "Payroll",
        "Research & Development", "Manufacturing Supplies", "Construction", "Logistics",
        "Customer Support", "Security Services", "Miscellaneous"
    ]
    
    return prompt_expert(
        action_context=action_context,
        description_of_expert="A senior financial analyst with deep expertise in corporate spending categorization.",
        prompt=f"Given the following description: '{description}', classify the expense into one of these categories:\n{categories}"
    )
Step 2: Creating the Purchasing Rules Expert
To ensure that our purchasing rules expert is always using the latest policy guidelines, we will implement it to read the purchasing rules from a file on disk before validating an invoice. This approach makes our system dynamic—we don’t need to hardcode the rules into the agent, and they can be updated easily without modifying the code. Also, we don’t need a special format for the rules, the human-readable policy becomes the logic!

First, let’s assume that the purchasing rules are stored in a simple text file located at "config/purchasing_rules.txt". The expert will read these rules, insert them into the prompt, and then validate the invoice accordingly.

@register_tool(tags=["invoice_processing", "validation"])
def check_purchasing_rules(action_context: ActionContext, invoice_data: dict) -> dict:
    """
    Validate an invoice against company purchasing policies.
    
    Args:
        invoice_data: Extracted invoice details, including vendor, amount, and line items.
        
    Returns:
        A dictionary indicating whether the invoice is compliant, with explanations.
    """
    # Load the latest purchasing rules from disk
    rules_path = "config/purchasing_rules.txt"
    
    try:
        with open(rules_path, "r") as f:
            purchasing_rules = f.read()
    except FileNotFoundError:
        purchasing_rules = "No rules available. Assume all invoices are compliant."

    return prompt_expert(
        action_context=action_context,
        description_of_expert="A corporate procurement compliance officer with extensive knowledge of purchasing policies.",
        prompt=f"""
        Given this invoice data: {invoice_data}, check whether it complies with company purchasing rules.
        The latest purchasing rules are as follows:
        
        {purchasing_rules}
        
        Identify any violations or missing requirements. Respond with:
        - "compliant": true or false
        - "issues": A brief explanation of any problems found
        """
    )
How This Works
Reads the Rules from Disk:

Before validating the invoice, the tool opens and reads the purchasing rules from config/purchasing_rules.txt.
If the file is missing, it defaults to assuming all invoices are compliant (which can be changed based on business needs).
Inserts Rules into the Prompt:

The LLM expert receives the full policy text within the prompt to make informed decisions about compliance.
Returns a Validation Report:

The expert outputs whether the invoice passes or fails compliance and provides a brief explanation of any issues.
Example: Updating Purchasing Rules Without Code Changes
Let’s assume our purchasing rules file contains:

1. All purchases over $5,000 require pre-approval.
2. IT equipment purchases must be from approved vendors.
3. Travel expenses must include a justification.
4. Consulting fees over $10,000 require an SOW (Statement of Work).
If an invoice for $6,000 arrives without pre-approval, the agent will detect the violation automatically.

Example Invoice Data

{
    "invoice_number": "7890",
    "date": "2025-03-10",
    "vendor": "Tech Solutions Inc.",
    "total_amount": 6000,
    "line_items": [
        {"description": "High-end workstation", "quantity": 1, "total": 6000}
    ]
}
Expert Output

{
    "compliant": false,
    "issues": "This purchase exceeds $5,000 but lacks required pre-approval."
}
Generating Structured Responses Rather than Free Text

While this free-text reasoning is powerful, it isn’t always enough. There are situations where the results of an evaluation need to be captured in a format that other systems can work with. If an organization wants to log purchasing violations in a database, trigger automated workflows based on compliance status, or generate structured reports on spending patterns, then a loosely written explanation from an LLM—however insightful—won’t be enough. The output needs to follow a predictable structure so that it can be systematically processed.

To make this possible, we can shift from a purely natural language response to a structured JSON format. Instead of returning an open-ended explanation, we direct the LLM to produce a response that includes two key elements: a boolean flag that indicates whether the invoice is compliant (true or false), and a structured field that provides a justification for the decision. By using prompt_llm_for_json, we ensure that the response conforms to this expected format, eliminating ambiguity and making it easy to integrate with downstream systems.

The modification to our validation tool is straightforward. Instead of returning a free-text analysis, we define a schema that enforces the structured response. The expert tool reads the purchasing rules from disk, provides those rules as context to the LLM, and requests a JSON-formatted response that aligns with the schema.

@register_tool(tags=["invoice_processing", "validation"])
def check_purchasing_rules(action_context: ActionContext, invoice_data: dict) -> dict:
    """
    Validate an invoice against company purchasing policies, returning a structured response.
    
    Args:
        invoice_data: Extracted invoice details, including vendor, amount, and line items.
        
    Returns:
        A structured JSON response indicating whether the invoice is compliant and why.
    """
    rules_path = "config/purchasing_rules.txt"

    try:
        with open(rules_path, "r") as f:
            purchasing_rules = f.read()
    except FileNotFoundError:
        purchasing_rules = "No rules available. Assume all invoices are compliant."

    validation_schema = {
        "type": "object",
        "properties": {
            "compliant": {"type": "boolean"},
            "issues": {"type": "string"}
        }
    }

    return prompt_llm_for_json(
        action_context=action_context,
        schema=validation_schema,
        prompt=f"""
        Given this invoice data: {invoice_data}, check whether it complies with company purchasing rules.
        The latest purchasing rules are as follows:
        
        {purchasing_rules}
        
        Respond with a JSON object containing:
        - `compliant`: true if the invoice follows all policies, false otherwise.
        - `issues`: A brief explanation of any violations or missing requirements.
        """
    )
Step 3: Updating the Invoice Processing Agent
Now that we have our expert tools, we update the Invoice Processing Agent to intelligently decide:

When to extract invoice data
When to categorize expenditures
When to validate invoices
How to store the processed information

Full Agent Code

def create_invoice_agent():
    # Create action registry with invoice tools
    action_registry = PythonActionRegistry()

    # Define invoice processing goals
    goals = [
        Goal(
            name="Persona",
            description="You are an Invoice Processing Agent, specialized in handling invoices efficiently."
        ),
        Goal(
            name="Process Invoices",
            description="""
            Your goal is to process invoices accurately. For each invoice:
            1. Extract key details such as vendor, amount, and line items.
            2. Generate a one-sentence summary of the expenditure.
            3. Categorize the expenditure using an expert.
            4. Validate the invoice against purchasing policies.
            5. Store the processed invoice with categorization and validation status.
            6. Return a summary of the invoice processing results.
            """
        )
    ]

    # Define agent environment
    environment = PythonEnvironment()

    return Agent(
        goals=goals,
        agent_language=AgentFunctionCallingActionLanguage(),
        action_registry=action_registry,
        generate_response=generate_response,
        environment=environment
    )
Step 4: Testing the New Capabilities
With the new expert functionality, our agent can now process, categorize, and validate invoices dynamically. Here’s how we can use it:

invoice_text = """
    Invoice #4567
    Date: 2025-02-01
    Vendor: Tech Solutions Inc.
    Items: 
      - Laptop - $1,200
      - External Monitor - $300
    Total: $1,500
"""

# Create an agent instance
agent = create_invoice_agent()

# Process the invoice
response = agent.run(f"Process this invoice:\n\n{invoice_text}")

print(response)
Expected Output:

Invoice #4567
- Categorized as: IT Equipment
- Compliance Check: Passed
- Stored successfully

AI Agents and Agentic AI Architecture in Python

Module 1

Using Human Policies for Document-as-Implementation

The Document-as-Implementation Pattern
In the last lesson, we did something new. We had the agent read some human knowledge at startup to understand what to do. This seemingly simple pattern - loading a purchasing policy from disk and letting the AI interpret it - represents a significant paradigm shift in how we build intelligent systems.

From Hardcoded Logic to Document-as-Implementation
Traditionally, even in AI systems, business logic follows a familiar pattern: developers analyze requirements, understand policies, and then translate them into code. This creates several fundamental challenges:

Translation Loss: Each time human knowledge moves from one format to another (from policy document → to developer understanding → to code implementation), some context and nuance is inevitably lost.

Maintenance Challenges: When policies change, a document gets updated, but the code implementing those policies remains unchanged until a developer manually updates it.

Knowledge Silos: Only developers can modify the system’s behavior, creating bottlenecks and dependencies.

Validation Difficulties: It becomes hard to verify if the code accurately implements the current policies.

The Document-as-Implementation Pattern
Let’s examine what we did in the invoice processing example:

# Load the latest purchasing rules from disk
rules_path = "config/purchasing_rules.txt"

try:
    with open(rules_path, "r") as f:
        purchasing_rules = f.read()
except FileNotFoundError:
    purchasing_rules = "No rules available. Assume all invoices are compliant."
This simple file-loading operation represents a profound shift in how we implement business logic:

Key Benefits
1. Direct Use of Human Knowledge
Instead of translating policies into code, we’re using the human-readable document directly as the implementation. The document itself becomes the logic.

2. Dynamic Updates Without Code Changes
When a policy changes, simply update the document. The system automatically incorporates the latest rules on the next run without requiring developer intervention or redeployment.

3. Organizational Alignment
This pattern aligns with how organizations naturally work. The people responsible for defining policies (legal, compliance, finance) can directly maintain the documents that drive system behavior.

4. Traceability and Transparency
When the agent makes a decision, it’s clear which document version it used. This creates an auditable trail that connects decisions to the exact policies in effect at that time.

Beyond Basic File Loading
This pattern can be extended far beyond simple text files:

Knowledge Bases and Wikis
Agents can pull from entire knowledge bases, internal wikis, or document management systems, allowing them to reason over vast organizational knowledge.

Versioned Documents
By using versioned documents, you can trace exactly which version of a policy was applied for any decision, critical for compliance and auditing.

Multi-Document Reasoning
Agents can reason across multiple documents, reconciling information from different sources just as a human expert would.

Document-Guided Workflows
Process documentation can drive entire workflows, with the agent determining next steps based on standard operating procedures.

Real-World Applications
This pattern is powerful across many domains:

Compliance Systems
Financial institutions can keep regulatory compliance documents as the source of truth, ensuring AI systems always follow current regulations.

Healthcare Protocols
Medical systems can reason directly from clinical guidelines and hospital protocols, adapting immediately when best practices are updated.

HR Policy Enforcement
HR systems can evaluate requests against the actual policy documents that human resources maintains.

Customer Service
Support systems can use product documentation, FAQs, and policy documents to provide consistent, accurate responses.

Implementing This Pattern
To implement this pattern effectively:

Identify human-readable documents that contain decision-making knowledge in your organization

Treat these documents as interfaces between human experts and AI systems

Create loading mechanisms that pull current document versions at runtime

Design prompts that effectively frame the document content for the LLM to reason over

Structure outputs to capture decisions in formats that downstream systems can process

Conclusion
The document-as-implementation pattern fundamentally changes our approach to building intelligent systems. Instead of translating human knowledge into code, we allow AI to work directly with the documents humans create and maintain.

This creates a more maintainable, transparent system where:

Domain experts can directly influence system behavior
Changes in policy immediately affect system decisions
The gap between documentation and implementation disappears
Organizations can leverage their existing knowledge artifacts



Persona & Self-Prompting Review
Graded Assignment • 30 min

Using ChatGPT or Claude to assist you, write 2-3 paragraphs discussing why Personas are an efficient prompting abstraction.


Personas are an efficient prompting abstraction because they encapsulate a consistent role, tone, and domain-specific behavior that guide large language models (LLMs) like ChatGPT to generate more accurate, relevant, and context-aware responses. By defining a persona—such as a "technical recruiter," "medical advisor," or "financial analyst"—users can reduce the need for repetitive prompt instructions while ensuring responses align with the expectations of a specific audience or task. This abstraction simplifies interactions and enables users to focus on outcomes rather than prompt engineering details.

Additionally, personas enhance multi-turn conversations and agent-based workflows by maintaining coherence in role and objectives. For example, in Agentic AI systems, each agent may be assigned a distinct persona with predefined expertise and decision-making criteria. This allows for better coordination, delegation, and task-specific reasoning across agents. As a result, personas not only improve prompt consistency but also contribute to modular, reusable, and scalable prompt designs for more complex applications like RAG (Retrieval-Augmented Generation), autonomous research agents, or customer service bots.

Question 2
Using ChatGPT or Claude to help you, write 2-3 paragraphs discussing why exposing self-prompting as a tool helps improve modularity, testability, and other properties of the agent versus integrating the prompt directly into the agent reasoning.

Exposing self-prompting as a tool—rather than embedding it directly into the core reasoning loop of an agent—enhances modularity by separating prompt engineering from the agent's primary decision-making logic. This decoupling allows developers to iterate and optimize prompts independently of the broader agent framework, making it easier to update, swap, or A/B test prompts without affecting the core agent architecture. Each prompt can be encapsulated as a discrete function or callable component, fostering composability across different tasks and enabling prompt reuse across various agents or workflows.

In terms of testability, treating self-prompting as an external tool makes the system more transparent and predictable. Each prompt can be tested in isolation with known inputs and expected outputs, enabling rigorous unit testing and prompt evaluation. This separation also supports more deterministic debugging and logging, which is crucial in diagnosing errors in reasoning chains. Furthermore, exposing prompting as a tool aligns with the principle of separation of concerns, enhancing maintainability, traceability, and observability of agent behavior. This design pattern supports more agentic architectures, where agents can learn to reason about and select their own prompts dynamically, empowering more robust meta-cognition and adaptability.


What is meta-cognition with an example?



Meta-cognition refers to “thinking about thinking.” In the context of AI agents (or even humans), it’s the ability to monitor, evaluate, and adjust one’s own reasoning process to achieve a goal more effectively.

Simple Example in Humans:
A student solving a math problem might pause and think:
“Wait, this method isn’t working—maybe I should try a different formula.”
Here, the student isn’t just solving the problem—they’re evaluating their own strategy and adjusting it. That’s meta-cognition.

Example in AI Agents:
An agent is tasked with answering a complex research question. It tries one approach using a search query and retrieves documents. Then it “thinks”:

“The documents I found are too general. I should reformulate my query to be more specific to the target domain.”

Here, the agent is reflecting on the effectiveness of its reasoning or tool use, then adapting its behavior—this is meta-cognition.

In systems with self-prompting, meta-cognition might look like the agent evaluating its own prompt and saying:

“This prompt is too vague to generate a useful response. Let me refine it before continuing.”

By building meta-cognitive capabilities into agents, we move closer to autonomous systems that can self-correct, optimize, and reason more like humans.


AI Agents and Agentic AI Architecture in Python

Module 2

The MATE Design Principles for AI Agents

The MATE Design Principles
In chess, a checkmate represents the perfect execution of strategy - every piece positioned correctly, every move calculated for maximum effect, leading to an inevitable victory. When building AI agents, we can apply similar strategic thinking through the MATE principles: Model efficiency, Action specificity, Token efficiency, and Environmental safety.

Model Efficiency: Choose Your Pieces Wisely
In chess, each piece has unique capabilities. You wouldn’t use a queen when a pawn would suffice. Similarly, model efficiency means selecting the right LLM for each task:

@register_tool(description="Extract basic contact information from text")
def extract_contact_info(action_context: ActionContext, text: str) -> dict:
    """Extract name, email, and phone from text using a smaller, faster model."""
    # Use a smaller model for simple extraction
    response = action_context.get("fast_llm")(Prompt(messages=[
        {"role": "system", "content": "Extract contact information in JSON format."},
        {"role": "user", "content": text}
    ]))
    return json.loads(response)

@register_tool(description="Analyze complex technical documentation")
def analyze_technical_doc(action_context: ActionContext, document: str) -> dict:
    """Perform deep analysis of technical documentation."""
    # Use a more capable model for complex analysis
    response = action_context.get("powerful_llm")(Prompt(messages=[
        {"role": "system", "content": "Analyze technical this documentation thoroughly to identify potential contradictions in process that could lead to unexpected problems."},
        {"role": "user", "content": document}
    ]))
    return json.loads(response)
Action Specificity: Control the Board
In chess, precise positioning limits your opponent’s options. Similarly, specific actions limit the potential for misuse or confusion:

# Too generic - opens up possibilities for misuse
@register_tool(description="Modify calendar events")
def update_calendar(action_context: ActionContext, 
                   event_id: str,
                   updates: dict) -> dict:
    """Update any aspect of a calendar event."""
    return calendar.update_event(event_id, updates)

# More specific - clear purpose and limited scope
@register_tool(description="Reschedule a meeting you own to a new time")
def reschedule_my_meeting(action_context: ActionContext,
                         event_id: str,
                         new_start_time: str,
                         new_duration_minutes: int) -> dict:
    """
    Reschedule a meeting you own to a new time.
    Only works for meetings where you are the organizer.
    """
    # Verify ownership
    event = calendar.get_event(event_id)
    if event.organizer != action_context.get("user_email"):
        raise ValueError("Can only reschedule meetings you organize")
        
    # Validate new time is in the future
    new_start = datetime.fromisoformat(new_start_time)
    if new_start < datetime.now():
        raise ValueError("Cannot schedule meetings in the past")
        
    return calendar.update_event_time(
        event_id,
        new_start_time=new_start_time,
        duration_minutes=new_duration_minutes
    )
Token Efficiency: Maximize Every Move

In chess, every move should advance your position. Let’s imagine that we need to analyze some sales data and all we need is the YoY growth and top 3 trends. If we prompt for more than we need or provide unnecessary context, we’re wasting tokens. Sometimes, we might need to add more tokens to get the appropriate level of reasoning, but we should test and optimize for the most efficient use of tokens. A longer prompt will consume more input tokens and if we assign the LLM more to do than we need, we also will end up with wasted output tokens:

# Token inefficient - includes unnecessary context
@register_tool(description="Analyze sales data to identify trends and patterns...")
def analyze_sales(action_context: ActionContext, data: str) -> str:
    """
    This function will analyze sales data to identify trends and patterns.
    It looks at various aspects including:
    - Monthly trends
    - Seasonal patterns
    - Year-over-year growth
    - Product category performance
    - Regional variations
    - Customer segments
    
    The analysis will be thorough and consider multiple factors...
    [More verbose documentation]
    """
    
    # This prompt asks for unneeded analysis, leading to both a waste of input and output tokens
    return prompt_llm(action_context, f"""
        Analyze this sales data thoroughly. Consider monthly trends,
        seasonal patterns, year-over-year growth, product categories,
        regional variations, and customer segments. Provide detailed
        insights about all these aspects.
        
        Data: {data}
        
        Please give a comprehensive analysis...
    """)

# Token efficient - focused and precise
@register_tool(description="Analyze sales data for key trends")
def analyze_sales(action_context: ActionContext, data: str) -> str:
    """Calculate key sales metrics and identify significant trends."""
    
    # This prompt is focused and precise, using tokens efficiently
    return prompt_llm(action_context, f"""
        Sales Data: {data}
        1. Calculate YoY growth
        2. Identify top 3 trends
        3. Flag significant anomalies
    """)



async

 async def execute(self):
        """Execute all actions in the transaction."""
        try:
            for action, args in self.actions:
                result = action.run(**args)
                self.executed.append(action)
        except Exception as e:
            # If any action fails, reverse everything done so far
            await self.rollback()
            raise e


This keyword marks the function as asynchronous, meaning it can use await inside the function to perform non-blocking I/O operations (like HTTP requests, database queries, or file access).

It is useful when you want your program to continue doing other tasks while waiting for something to complete.


🔁 Common Usage
A method like this is typically found in classes that need to perform asynchronous tasks, such as:

Running a background job

Calling APIs

Interacting with a database

Coordinating tasks in an Agent or RAG system

AI Agents and Agentic AI Architecture in Python

Module 2

AI Agents & Environment Safety

Environmental Safety for AI Agents
When agents take actions in the real world, we need robust safety mechanisms. Let’s explore patterns for building safe action systems using a calendar coordination example.

Pattern 1: Reversible Actions
When agents interact with the real world, mistakes can happen. Even if an agent makes the right decision with the information it has, external circumstances might change that make an action undesirable after it’s taken. Reversible actions provide a safety net, allowing us to “undo” operations when things go wrong.

The key to making actions reversible is capturing enough information during the execution to fully reverse the operation later. This isn’t always straightforward - for example, deleting a calendar event is easy to reverse if you saved all the event details, but reversing an email that’s already been read requires careful thought about what “reversal” means in your system. Sometimes, the best we can do is send a follow-up correction.

Let’s examine how we might structure actions so they can be undone:

class ReversibleAction:
    def __init__(self, execute_func, reverse_func):
        self.execute = execute_func
        self.reverse = reverse_func
        self.execution_record = None

    def run(self, **args):
        """Execute action and record how to reverse it."""
        result = self.execute(**args)
        self.execution_record = {
            "args": args,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        return result

    def undo(self):
        """Reverse the action using recorded information."""
        if not self.execution_record:
            raise ValueError("No action to reverse")
        return self.reverse(**self.execution_record)

# Example using reversible actions
create_event = ReversibleAction(
    execute_func=calendar.create_event,
    reverse_func=lambda **record: calendar.delete_event(record["result"]["event_id"])
)

send_invite = ReversibleAction(
    execute_func=calendar.send_invite,
    reverse_func=lambda **record: calendar.cancel_invite(record["result"]["invite_id"])
)
Now, if we record a log of the agent’s actions, we can undo them if needed. The agent itself can also be given a “revert” tool to undo its most recent action or revert all actions under certain circumstance

Pattern 2: Transaction Management
Individual reversible actions are helpful, but real tasks often require multiple coordinated actions. For example, scheduling a meeting might involve checking availability, creating a calendar event, and sending notifications. Transaction management lets us treat these related actions as a single unit that either completely succeeds or completely fails.

The transaction pattern becomes particularly powerful when combined with reversible actions. If any step in a multi-step process fails, we can automatically roll back all previous steps. This prevents inconsistent states where, for example, a calendar event exists but notifications weren’t sent, or notifications were sent but the event wasn’t created. It’s like having an “undo” button for entire sequences of operations.

Here is a simple transaction management system that we might use to collect the actions that an agent is executing during the agent loop and then either commit or rollback all of them at the end of the loop.

class ActionTransaction:
    def __init__(self):
        self.actions = []
        self.executed = []
        self.committed = False
        self.transaction_id = str(uuid.uuid4())

    def add(self, action: ReversibleAction, **args):
        """Queue an action for execution."""
        if self.committed:
            raise ValueError("Transaction already committed")
        self.actions.append((action, args))

    async def execute(self):
        """Execute all actions in the transaction."""
        try:
            for action, args in self.actions:
                result = action.run(**args)
                self.executed.append(action)
        except Exception as e:
            # If any action fails, reverse everything done so far
            await self.rollback()
            raise e

    async def rollback(self):
        """Reverse all executed actions in reverse order."""
        for action in reversed(self.executed):
            await action.undo()
        self.executed = []

    def commit(self):
        """Mark transaction as committed."""
        self.committed = True
Pattern 3: Staged Execution with Review
Sometimes we want more control than just automatic rollback on failure. Staged execution allows us to plan out a series of actions, review them for safety and correctness, and only then execute them. This is particularly valuable when some actions are more reversible than others, or when we want to give a human or more capable AI system a chance to review plans before execution.

The review phase can catch problems that might not be apparent when looking at individual actions. For example, while each step in setting up a large meeting might be valid on its own, a reviewer might notice that the combination of actions would overwhelm a team’s schedule or violate company policies about meeting sizes. This pattern also allows for optimization - a reviewer might notice more efficient ways to accomplish the same goal.

Now let’s create a system that stages changes for review before execution:

Now let’s create a system that stages changes for review before execution:

class StagedActionEnvironment(Environment):
    def __init__(self):
        self.staged_transactions = {}
        self.llm = None  # High-capability LLM for review

    def stage_actions(self, task_id: str) -> ActionTransaction:
        """Create a new transaction for staging actions."""
        transaction = ActionTransaction()
        self.staged_transactions[task_id] = transaction
        return transaction

    def review_transaction(self, task_id: str) -> bool:
        """Have LLM review staged actions for safety."""
        transaction = self.staged_transactions.get(task_id)
        if not transaction:
            raise ValueError(f"No transaction found for task {task_id}")

        # Create a description of staged actions
        staged_actions = [
            f"Action: {action.__class__.__name__}\nArgs: {args}"
            for action, args in transaction.actions
        ]
        
        # The safest way to do this would be to send it for human review, but we can also imagine having a more capable AI system review it before the human to minimize the number of reviews that the human has to do. The more capable AI can review and reject potentially problematic actions earlier.
        
        review_prompt = f"""Review these staged actions for safety:
        
        Task ID: {task_id}
        
        Staged Actions:
        {staged_actions}
        
        Consider:
        1. Are all actions necessary for the task?
        2. Could any action have unintended consequences?
        3. Are the actions in a safe order?
        4. Is there a safer way to achieve the same goal?
        
        Should these actions be approved?
        """
        
        response = self.llm.generate(review_prompt)
        
        # If approved, notify the human and ask if
        # they want to proceed
        return "approved" in response.lower()

# Example usage:
async def schedule_team_meeting(env: StagedActionEnvironment, 
                              attendees: List[str],
                              duration: int):
    """Schedule a team meeting with safety checks."""
    task_id = str(uuid.uuid4())
    transaction = env.stage_actions(task_id)
    
    # Check availability (execute immediately)
    available_slots = calendar.check_availability(attendees, duration)
    if not available_slots:
        return {"error": "No available time slots"}
    
    best_slot = available_slots[0]
    
    # Stage the event creation
    transaction.add(create_event, 
                   title="Team Meeting",
                   time=best_slot,
                   duration=duration)
    
    # Draft email (execute immediately)
    email_draft = email.draft_message(
        to=attendees,
        subject="Team Meeting",
        body=f"Team meeting scheduled for {best_slot}"
    )
    
    # Stage the email send
    transaction.add(send_email, 
                   draft_id=email_draft.id)
    
    # Review staged actions...send to human review
    # or more capable AI for initial filtering
    if env.review_transaction(task_id):
        await transaction.execute()
        transaction.commit()
        return {"status": "scheduled"}
    else:
        return {"status": "rejected"}


Pattern 4: Single Safe Tool vs Multiple Risky Tools

At first glance, having many small, focused tools seems like good design - it follows the Unix philosophy of “do one thing and do it well.” However, when it comes to agent safety, this approach can backfire. Small, focused tools often lack the context to make safe decisions, and they put the burden of correct orchestration on the agent.

A single comprehensive tool, in contrast, can encapsulate not just the mechanics of an operation but also the business rules and safety constraints that make it correct. Instead of hoping the agent will check availability before scheduling a meeting, verify attendees before sending invites, and handle timezone conversions correctly, we can build all of these safety checks into a single tool. This makes it much harder for the agent to accidentally create problematic situations, even if it tries to use the tool in unexpected ways.

Let’s compare two approaches to meeting scheduling:

# Approach 1: Multiple loosely constrained tools
@register_tool(description="Create a calendar event")
def create_calendar_event(action_context: ActionContext,
                         title: str,
                         time: str,
                         attendees: List[str]) -> dict:
    """Create a calendar event."""
    return calendar.create_event(title=title,
                               time=time,
                               attendees=attendees)

@register_tool(description="Send email to attendees")
def send_email(action_context: ActionContext,
               to: List[str],
               subject: str,
               body: str) -> dict:
    """Send an email."""
    return email.send(to=to, subject=subject, body=body)

@register_tool(description="Update calendar event")
def update_event(action_context: ActionContext,
                 event_id: str,
                 updates: dict) -> dict:
    """Update any aspect of a calendar event."""
    return calendar.update_event(event_id, updates)

# Approach 2: Single comprehensive safe tool
@register_tool(description="Schedule a team meeting safely")
def schedule_team_meeting(action_context: ActionContext,
                         title: str,
                         description: str,
                         attendees: List[str],
                         duration_minutes: int,
                         timeframe: str = "next_week") -> dict:
    """
    Safely schedule a team meeting with all necessary coordination.
    
    This tool:
    1. Verifies all attendees are valid
    2. Checks calendar availability
    3. Creates the event at the best available time
    4. Sends appropriate notifications
    5. Handles all error cases
    """
    # Input validation
    if not 15 <= duration_minutes <= 120:
        raise ValueError("Meeting duration must be between 15 and 120 minutes")
    
    if len(attendees) > 10:
        raise ValueError("Cannot schedule meetings with more than 10 attendees")
        
    # Verify attendees
    valid_attendees = validate_attendees(attendees)
    if len(valid_attendees) != len(attendees):
        raise ValueError("Some attendees are invalid")
        
    # Find available times
    available_slots = find_available_times(
        attendees=valid_attendees,
        duration=duration_minutes,
        timeframe=timeframe
    )
    
    if not available_slots:
        return {
            "status": "no_availability",
            "message": "No suitable time slots found"
        }
    
    # Create event at best time
    event = calendar.create_event(
        title=title,
        description=description,
        time=available_slots[0],
        duration=duration_minutes,
        attendees=valid_attendees
    )

      notifications.send_meeting_scheduled(
        event_id=event.id,
        attendees=valid_attendees
    )
    
    return {
        "status": "scheduled",
        "event_id": event.id,
        "scheduled_time": available_slots[0]
    }
The single comprehensive tool is safer because:

It enforces proper sequencing of operations
It includes all necessary validation
It handles error cases consistently
It prevents misuse of individual operations
With the multiple tool approach, an agent might:

Create an event without checking availability
Send emails without creating the event
Update events without notifying attendees
Exceed reasonable size limits
The single tool encapsulates the business logic of what makes a “safe” meeting scheduling operation, making it much harder for the agent to create problematic situations.


AI Agents and Agentic AI Architecture in Python

Module 3

Building Multi-Agent Systems: Agent-to-Agent Communication


A call_agent Tool
Imagine we want to create a system where multiple specialized agents can work together, each bringing their unique capabilities to solve complex problems. For example, we might have a primary agent that coordinates high-level tasks but needs to delegate specialized work to other agents. To make this possible, we need a way for agents to communicate with each other.

As we have seen in the past, often the most effective way to add a capability to an agent is to expose it as a tool. This architectural approach makes it easy to create systems with multi-agent coordination by simply exposing the right tool interfaces between them. Let’s build a multi-agent collaboration capability by creating a call_agent tool that allows one agent to invoke another and receive its results. We’ll see how the ActionContext makes this surprisingly straightforward.

First, let’s examine how the tool will work within an agent’s execution:

@register_tool()
def call_agent(action_context: ActionContext, 
               agent_name: str, 
               task: str) -> dict:
    """
    Invoke another agent to perform a specific task.
    
    Args:
        action_context: Contains registry of available agents
        agent_name: Name of the agent to call
        task: The task to ask the agent to perform
        
    Returns:
        The result from the invoked agent's final memory
    """
    # Get the agent registry from our context
    agent_registry = action_context.get_agent_registry()
    if not agent_registry:
        raise ValueError("No agent registry found in context")
    
    # Get the agent's run function from the registry
    agent_run = agent_registry.get_agent(agent_name)
    if not agent_run:
        raise ValueError(f"Agent '{agent_name}' not found in registry")
    
    # Create a new memory instance for the invoked agent
    invoked_memory = Memory()
    
    try:
        # Run the agent with the provided task
        result_memory = agent_run(
            user_input=task,
            memory=invoked_memory,
            # Pass through any needed context properties
            action_context_props={
                'auth_token': action_context.get('auth_token'),
                'user_config': action_context.get('user_config'),
                # Don't pass agent_registry to prevent infinite recursion
            }
        )
        
        # Get the last memory item as the result
        if result_memory.items:
            last_memory = result_memory.items[-1]
            return {
                "success": True,
                "agent": agent_name,
                "result": last_memory.get("content", "No result content")
            }
        else:
            return {
                "success": False,
                "error": "Agent failed to run."
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
Let’s examine how the call_agent tool enables inter-agent communication within our multi-agent system. This tool follows a careful protocol to ensure proper agent invocation while maintaining appropriate boundaries. When one agent needs to leverage the capabilities of another, it calls this function with the target agent’s name and a description of the task to be performed. The function first retrieves the agent registry from the action context, which serves as a central directory of all available agents in the system. After confirming the requested agent exists, it creates a fresh memory instance for the agent being called, ensuring each invocation starts with a clean slate.


The core of this function lies in its actual invocation of the target agent. It calls the agent’s run function with the provided task, passing along only the necessary context properties from the calling agent. Notice how it explicitly avoids passing the entire agent registry to prevent infinite recursion loops that could occur if agents continuously call each other without proper constraints. After the invoked agent completes its task, the function extracts the final memory item as the result, which represents the agent’s ultimate conclusion or output. This structured approach to inter-agent communication ensures that agents can collaborate effectively while maintaining clear separation of concerns and avoiding potential pitfalls like circular references or excessive context sharing that could compromise the system’s stability or security.

Building a Meeting Scheduling System with Specialized Agents
Let’s look at what a project management system where two agents work together to identify and schedule necessary meetings might look like. The project management agent decides when meetings are needed, while a scheduling specialist handles the logistics of actually arranging them.

First, let’s look at the tools available to our scheduling specialist. This agent needs to interact with calendars and create invites:

@register_tool()
def check_availability(
    action_context: ActionContext,
    attendees: List[str],
    start_date: str,
    end_date: str,
    duration_minutes: int,
    _calendar_api_key: str
) -> List[Dict]:
    """Find available time slots for all attendees."""
    return calendar_service.find_available_slots(...)

@register_tool()
def create_calendar_invite(
    action_context: ActionContext,
    title: str,
    description: str,
    start_time: str,
    duration_minutes: int,
    attendees: List[str],
    _calendar_api_key: str
) -> Dict:
    """Create and send a calendar invitation."""
    return calendar_service.create_event(...)
The scheduling specialist is focused entirely on finding times and creating meetings:

scheduler_agent = Agent(
    goals=[
        Goal(
            name="schedule_meetings",
            description="""Schedule meetings efficiently by:
            1. Finding times that work for all attendees
            2. Creating and sending calendar invites
            3. Handling any scheduling conflicts"""
        )
    ],
...
)
Now let’s look at our project management agent. This agent focuses on project status and deciding when meetings are needed:

@register_tool()
def get_project_status(
    action_context: ActionContext,
    project_id: str,
    _project_api_token: str
) -> Dict:
    """Retrieve current project status information."""
    return project_service.get_status(...)

@register_tool()
def update_project_log(
    action_context: ActionContext,
    entry_type: str,
    description: str,
    _project_api_token: str
) -> Dict:
    """Record an update in the project log."""
    return project_service.log_update(...)

@register_tool()
def call_agent(
    action_context: ActionContext,
    agent_name: str,
    task: str
) -> Dict:
    """Delegate to a specialist agent."""
    # Implementation as shown in previous tutorial
The project management agent uses these tools to monitor progress and arrange meetings when needed:

project_manager = Agent(
    goals=[
        Goal(
            name="project_oversight",
            description="""Manage project progress by:
            1. Getting the current project status
            2. Identifying when meetings are needed if there are issues in the project status log
            3. Delegating meeting scheduling to the "scheduler_agent" to arrange the meeting
            4. Recording project updates and decisions"""
        )
    ],
    ...
)
This division of responsibilities keeps each agent focused on its core competency:

The project manager understands project status and when meetings are needed
The scheduler excels at finding available times and managing calendar logistics
The call_agent tool allows seamless collaboration between them
The call_agent Tool
The call_agent tool manages several important aspects of agent interaction:

Memory Isolation: Each invoked agent gets its own memory instance, preventing confusion between different agents’ conversation histories.

Context Management: We carefully control what context properties are passed to the invoked agent, preventing infinite recursion while ensuring necessary resources are available.

Result Handling: The tool extracts the final memory item as the result, providing a clean way to return information to the calling agent.

Registering Agents
To make this system work, we need to register our agents in the registry:

class AgentRegistry:
    def __init__(self):
        self.agents = {}
        
    def register_agent(self, name: str, run_function: callable):
        """Register an agent's run function."""
        self.agents[name] = run_function
        
    def get_agent(self, name: str) -> callable:
        """Get an agent's run function by name."""
        return self.agents.get(name)

# When setting up the system
registry = AgentRegistry()
registry.register_agent("scheduler_agent", scheduler_agent.run)

# Include registry in action context
action_context = ActionContext({
    'agent_registry': registry,
    # Other shared resources...
})
The ActionContext provides a clean way to make the agent registry available to the call_agent tool without exposing it directly to all tools. When an agent needs to delegate a task, it simply uses the tool like any other, and the environment system handles the details of finding and invoking the right agent.

This architecture allows us to build complex multi-agent systems where each agent maintains its specialization while being able to collaborate with other agents when needed. The memory isolation ensures that each agent works with a clean context, while the result extraction provides a standard way to pass information back to the calling agent.

AI Agents and Agentic AI Architecture in Python
Module 3
Agent Interaction Patterns with Memory

Memory Interaction Patterns in Multi-Agent Systems
When agents work together, how they share and manage memory dramatically affects their collaboration. Let’s explore different patterns for memory interaction between agents, understanding when each pattern is most useful and how to implement it.

Message Passing: The Basic Pattern
The simplest form of agent interaction is message passing, where one agent sends a request and receives a response. This is like sending an email to a colleague - they get your message, do some work, and send back their results. You don’t see how they arrived at their answer; you just get their final response.

Here’s how we implement basic message passing:

@register_tool()
def call_agent(action_context: ActionContext, 
               agent_name: str, 
               task: str) -> dict:
    """Basic message passing between agents."""
    agent_registry = action_context.get_agent_registry()
    agent_run = agent_registry.get_agent(agent_name)
    
    # Create fresh memory for the invoked agent
    invoked_memory = Memory()
    
    # Run agent and get result
    result_memory = agent_run(
        user_input=task,
        memory=invoked_memory
    )
    
    # Return only the final memory item
    return {
        "result": result_memory.items[-1].get("content", "No result")
    }
This pattern works well when the first agent only needs the final answer, not the reasoning process. For example, if a project manager agent asks a scheduling agent to find a meeting time, it might only need to know when the meeting was scheduled, not how the time was chosen.

Memory Reflection: Learning from the Process
Sometimes we want the first agent to understand how the second agent reached its conclusion. This is like asking a colleague to not just give you their answer, but to explain their thought process. We can achieve this by copying all of the second agent’s memories back to the first agent:

@register_tool()
def call_agent_with_reflection(action_context: ActionContext, 
                             agent_name: str, 
                             task: str) -> dict:
    """Call agent and receive their full thought process."""
    agent_registry = action_context.get_agent_registry()
    agent_run = agent_registry.get_agent(agent_name)
    
    # Create fresh memory for invoked agent
    invoked_memory = Memory()
    
    # Run agent
    result_memory = agent_run(
        user_input=task,
        memory=invoked_memory
    )
    
    # Get the caller's memory
    caller_memory = action_context.get_memory()
    
    # Add all memories from invoked agent to caller
    # although we could leave off the last memory to
    # avoid duplication
    for memory_item in result_memory.items:
        caller_memory.add_memory({
            "type": f"{agent_name}_thought",  # Mark source of memory
            "content": memory_item["content"]
        })
    
    return {
        "result": result_memory.items[-1].get("content", "No result"),
        "memories_added": len(result_memory.items)
    }
This pattern is valuable when the first agent needs to understand the reasoning process. For instance, if a research coordinator agent asks a data analysis agent to study some results, seeing the analysis process helps the coordinator better understand and use the conclusions.

Memory Handoff: Continuing the Conversation
Sometimes we want the second agent to pick up where the first agent left off, with full context of what’s happened so far. This is like having a colleague step in to take over a project - they need to know everything that’s happened up to that point:

@register_tool()
def hand_off_to_agent(action_context: ActionContext, 
                      agent_name: str, 
                      task: str) -> dict:
    """Transfer control to another agent with shared memory."""
    agent_registry = action_context.get_agent_registry()
    agent_run = agent_registry.get_agent(agent_name)
    
    # Get the current memory to hand off
    current_memory = action_context.get_memory()
    
    # Run agent with existing memory
    result_memory = agent_run(
        user_input=task,
        memory=current_memory  # Pass the existing memory
    )
    
    return {
        "result": result_memory.items[-1].get("content", "No result"),
        "memory_id": id(result_memory)
    }
This pattern is useful for complex tasks where context is crucial. For example, if a customer service agent hands off to a technical support agent, the technical agent needs to know the full history of the customer’s issue.


AI Agents and Agentic AI Architecture in Python
Module 3
Advanced Agent Interaction


Selective Memory Sharing: Using LLM Understanding for Context Selection
Sometimes we want an agent to intelligently choose which parts of its memory to share with another agent. Instead of using rigid rules, we can leverage the LLM’s understanding of context to select the most relevant memories for the task at hand.

Let’s implement a version of memory sharing that uses the LLM to analyze and select relevant memories with self-prompting:

@register_tool(description="Delegate a task to another agent with selected context")
def call_agent_with_selected_context(action_context: ActionContext,
                                   agent_name: str,
                                   task: str) -> dict:
    """Call agent with LLM-selected relevant memories."""
    agent_registry = action_context.get_agent_registry()
    agent_run = agent_registry.get_agent(agent_name)
    
    # Get current memory and add IDs
    current_memory = action_context.get_memory()
    memory_with_ids = []
    for idx, item in enumerate(current_memory.items):
        memory_with_ids.append({
            **item,
            "memory_id": f"mem_{idx}"
        })
    
    # Create schema for memory selection
    selection_schema = {
        "type": "object",
        "properties": {
            "selected_memories": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "ID of a memory to include"
                }
            },
            "reasoning": {
                "type": "string",
                "description": "Explanation of why these memories were selected"
            }
        },
        "required": ["selected_memories", "reasoning"]
    }
    
    # Format memories for LLM review
    memory_text = "\n".join([
        f"Memory {m['memory_id']}: {m['content']}" 
        for m in memory_with_ids
    ])
    
    # Ask LLM to select relevant memories
    selection_prompt = f"""Review these memories and select the ones relevant for this task:

Task: {task}

Available Memories:
{memory_text}

Select memories that provide important context or information for this specific task.
Explain your selection process."""

    # Self-prompting magic to find the most relevant memories
    selection = prompt_llm_for_json(
        action_context=action_context,
        schema=selection_schema,
        prompt=selection_prompt
    )
    

 # Create filtered memory from selection
    filtered_memory = Memory()
    selected_ids = set(selection["selected_memories"])
    for item in memory_with_ids:
        if item["memory_id"] in selected_ids:
            # Remove the temporary memory_id before adding
            item_copy = item.copy()
            del item_copy["memory_id"]
            filtered_memory.add_memory(item_copy)
    
    # Run the agent with selected memories
    result_memory = agent_run(
        user_input=task,
        memory=filtered_memory
    )
    
    # Add results and selection reasoning to original memory
    current_memory.add_memory({
        "type": "system",
        "content": f"Memory selection reasoning: {selection['reasoning']}"
    })
    
    for memory_item in result_memory.items:
        current_memory.add_memory(memory_item)
    
    return {
        "result": result_memory.items[-1].get("content", "No result"),
        "shared_memories": len(filtered_memory.items),
        "selection_reasoning": selection["reasoning"]
    }
This implementation makes memory selection more intelligent and transparent:

Each memory gets assigned a unique ID for reference.

The complete set of memories is presented to the LLM with their IDs.

The LLM analyzes the memories in the context of the specific task and selects the relevant ones using structured JSON output.

The LLM provides reasoning for its selection, which is preserved in the original agent’s memory.

For example, if a project management agent is delegating a budget review task, the interaction might look like this:

# Example memory contents:
memories = [
    {"type": "user", "content": "We need to build a new reporting dashboard"},
    {"type": "assistant", "content": "Initial cost estimate: $50,000"},
    {"type": "user", "content": "That seems high"},
    {"type": "assistant", "content": "Breakdown: $20k development, $15k design..."},
    {"type": "system", "content": "Project deadline updated to Q3"},
    {"type": "user", "content": "Can we reduce the cost?"}
]

# LLM's selection might return:
{
    "selected_memories": ["mem_1", "mem_3", "mem_5"],
    "reasoning": "Selected memories containing cost information and the request for cost reduction, excluding project timeline and general discussion as they're not directly relevant to the budget review task."
}
The second agent then receives only the memories about cost estimates, breakdowns, and the request for reduction, giving it focused context for its budget review task without extraneous information about timelines or other project aspects.


This approach has several advantages over rule-based filtering:

The selection process can understand context and implications, not just match patterns.

The reasoning is preserved, helping track why certain information was or wasn’t shared.

The selection can adapt to different types of tasks and contexts without changing the code.

The original agent maintains a record of what information was shared and why.

This pattern is valuable when you want to provide specific context without overwhelming the second agent with irrelevant information. For example, if a project planning agent asks a budget specialist to review costs, it might share only the memories related to resource allocation and expenses, not the entire project history.

Recap of the Four Memory Sharing Patterns
Each of these patterns serves a different purpose in agent collaboration:

Message passing keeps interactions simple and focused
Memory reflection helps agents learn from each other’s processes
Memory handoff enables seamless continuation of complex tasks
Selective memory sharing provides relevant context while reducing noise
The choice of pattern depends on your specific needs:

How much context does the second agent need?
Does the first agent need to understand the second agent’s process?
Should the conversation history be preserved?
Is there sensitive information that should be filtered?
By understanding these patterns, you can design agent interactions that effectively balance information sharing with task focus, leading to more efficient and capable multi-agent systems.


Write 2-3 paragraphs discussing the key patterns for agent interaction and when you might want to use each.


ChatGPT said:
Agent interaction patterns define how autonomous AI agents communicate, collaborate, and coordinate to achieve tasks. Three key patterns commonly observed in agentic systems include sequential, parallel, and hierarchical (or manager-worker) interactions.

In a sequential pattern, agents operate one after another, passing the output of one agent as the input to the next. This pattern is useful when tasks have clear dependencies or must follow a logical progression—such as a research agent gathering data, followed by a summarization agent, and finally a QA agent validating results. Parallel patterns, in contrast, involve agents working independently on different aspects of a task simultaneously. This is ideal for scenarios like document classification or multi-topic analysis, where subtasks are distinct and can be processed concurrently to improve efficiency.

The hierarchical (manager-worker) pattern introduces a manager agent that delegates tasks to specialized worker agents based on context or skill. This approach is powerful for complex, multi-faceted workflows such as customer support automation or enterprise decision-making, where task decomposition, prioritization, and coordination are crucial. Choosing the right pattern depends on task structure, interdependencies, performance requirements, and the level of autonomy desired for each agent.

Claude Said:

Agent interaction patterns have evolved to address different computational needs and coordination challenges. Sequential patterns work best for linear workflows where each agent builds upon the previous agent's output, such as a content creation pipeline where one agent researches, another writes, and a third edits. This pattern ensures quality control and specialization but can be slower due to its serial nature. Parallel patterns excel when multiple independent tasks can be executed simultaneously, like having different agents analyze various aspects of a dataset or gather information from multiple sources concurrently, significantly reducing overall processing time.
Hierarchical patterns prove valuable for complex problems requiring delegation and coordination, where a supervisor agent breaks down tasks and assigns them to specialized worker agents, then synthesizes their results. This mirrors traditional organizational structures and scales well for large, multi-faceted problems. Collaborative patterns emerge when agents need to negotiate, debate, or iteratively refine solutions together, such as in multi-agent reasoning scenarios where different agents represent different perspectives or expertise areas, ultimately converging on better solutions through interaction.
The choice of pattern depends heavily on your specific requirements: use sequential for quality-critical workflows, parallel for speed and independent subtasks, hierarchical for complex coordination needs, and collaborative for problems benefiting from multiple viewpoints or iterative refinement. Consider factors like latency requirements, the interdependence of subtasks, the need for quality control, and whether the problem benefits from diverse perspectives when selecting your interaction pattern.


AI Agents and Agentic AI Architecture in Python
Module 4
Clean AI Tools with Dependency Injection

Decoupling Tools from Agent and Other Dependencies
In the lessons, we have been writing tools that use an architectural concept that we haven’t discussed yet, the action_context. This pattern is helpful for supporting dependency injection and decoupling tools from the agent and other dependencies. Let’s explore how the action_context pattern works and why it’s valuable for building flexible and maintainable agent architectures.

Let’s take a look at an example of why we need dependency injection and how the action_context pattern can help. Consider building an AI agent that helps develop and review code. In a typical workflow, the agent first acts as an expert developer to write code based on requirements. Later, we want the agent to switch roles and act as a code reviewer, critiquing its own work. This seems straightforward at first, but it surfaces a crucial architectural challenge.

For the code review to be effective, the LLM needs to understand the full context of how the code was developed. What requirements were specified? What constraints were discussed? What alternatives were considered and rejected?

Let’s assume that all of this information exists in the conversation history stored in the agent’s memory. What we would like to do is allow the LLM to see the memory of what happened in the agent loop, which contains a history of the code generation and requirements, and use it to provide a more informed code review. Let’s attempt to implement this code review tool:

@register_tool(
    description="Analyze code quality and suggest improvements",
    tags=["code_quality"]
)
def analyze_code_quality(code: str) -> str:
    """Review code quality and suggest improvements."""
    # But how do we access the conversation history?
    # We can't just import the agent instance - that would create tight coupling
    
    return prompt_expert(
        description_of_expert="""
        Senior software architect reviewing code quality
        """,
        prompt=f"Review this code:\n{code}"
    )
We face an immediate problem. Our tool needs access to the conversation history in the memory, but we can’t simply import the agent instance or directly access its memory. That would create tight coupling between the tools and the agent implementation, making our tools less reusable and harder to test.

What we need is a way to provide the tool with access to memory and other resources without creating direct dependencies. This is where the ActionContext pattern comes in.

The ActionContext serves as a container for any resources that tools might need during their execution:

class ActionContext:
    def __init__(self, properties: Dict=None):
        self.context_id = str(uuid.uuid4())
        self.properties = properties or {}

    def get(self, key: str, default=None):
        return self.properties.get(key, default)

    def get_memory(self):
        return self.properties.get("memory", None)
Now we can modify our tool to accept an ActionContext, allowing it to access the conversation history without knowing where it comes from:

@register_tool(
    description="Analyze code quality and suggest improvements",
    tags=["code_quality"]
)
def analyze_code_quality(action_context: ActionContext, code: str) -> str:
    """Review code quality and suggest improvements."""
    # Get memory to understand the code's context
    memory = action_context.get_memory()
    
    # Extract relevant history
    development_context = []
    for mem in memory.get_memories():
        if mem["type"] == "user":
            development_context.append(f"User: {mem['content']}")
        # Hypotethical scenario where our agent includes the phrase "Here's the implementation" when it generates code
        elif mem["type"] == "assistant" and "Here's the implementation" in mem["content"]:
            development_context.append(f"Implementation Decision: {mem['content']}")
    
    # Create review prompt with full context
    review_prompt = f"""Review this code in the context of its development history:

Development History:
{'\n'.join(development_context)}

Current Implementation:
{code}

Analyze:
1. Does the implementation meet all stated requirements?
2. Are all constraints and considerations from the discussion addressed?
3. Have any requirements or constraints been overlooked?
4. What improvements could make the code better while staying within the discussed parameters?
"""
    
    generate_response = action_context.get("llm")
    return generate_response(review_prompt)
Handling Session or Request-Specific Dependencies
This pattern of dependency injection through ActionContext becomes even more valuable when we need to handle authentication-specific information. Consider a tool that needs to update a project management system with the status of code reviews. This requires authentication, but we don’t want to hardcode credentials or configure them at the tool level.

Here’s how we can use ActionContext to handle authentication:

@register_tool(
    description="Update code review status in project management system",
    tags=["project_management"]
)
def update_review_status(action_context: ActionContext, 
                        review_id: str, 
                        status: str) -> dict:
    """Update the status of a code review in the project system."""
    # Get the authentication token for this specific request
    auth_token = action_context.get("auth_token")
    if not auth_token:
        raise ValueError("Authentication token not found in context")
    
    # Make authenticated request
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        f"https://...someapi.../reviews/{review_id}/status",
        headers=headers,
        json={"status": status}
    )
    
    if response.status_code != 200:
        raise ValueError(f"Failed to update review status: {response.text}")
        
    return {"status": "updated", "review_id": review_id}
When the agent runs a tool, it provides all necessary context through the ActionContext:

def run(self, user_input: str, memory=None, action_context_props=None):
    """Execute the agent loop."""
    memory = memory or Memory()
    
    # Create context with all necessary resources
    action_context = ActionContext({
        'memory': memory,
        'llm': self.generate_response,
        # Request-specific auth
        **action_context_props
    })
    
    while True:
        prompt = self.construct_prompt(action_context, self.goals, memory)
        response = self.prompt_llm_for_action(action_context, prompt)
        result = self.handle_agent_response(action_context, response)
        
        if self.should_terminate(action_context, response):
            break

    # Run the agent and create custom context for the action to 
# pass to tools that need it
some_agent.run("Update the project status...", 
               memory=..., 
               # Pass request-specific auth token
               action_context_props={"auth_token": "my_auth_token"})
By using ActionContext, we’ve solved several key challenges:

Tools can access conversation history without being coupled to the agent implementation
Authentication and other request-specific information can be injected where needed
Tools remain independent and testable since their dependencies are explicitly declared
The agent can provide different contexts for different execution environments (development, production, testing)

AI Agents and Agentic AI Architecture in Python
Module 4
Clean Tool Dependency Injection with the Environment

Dependency Injection, the Environment, and the Decorator
Now that we have our ActionContext to hold shared resources and dependencies, we face a new challenge: how do we get it to just the tools that need it? We need a solution that provides dependencies selectively, giving each tool access to only the resources it requires.

Consider that many tools are simple and self-contained, needing only their explicit parameters. A basic string manipulation tool shouldn’t receive memory access or authentication tokens it doesn’t use. Not only would this add unnecessary complexity, but it could also create security concerns by exposing sensitive information to tools that don’t need it.

For example, a simple text formatting tool needs nothing beyond its input text:

@register_tool(description="Convert text to uppercase")
def to_uppercase(text: str) -> str:
    """Convert input text to uppercase."""
    return text.upper()
While a tool that interacts with external services needs authentication:

@register_tool(description="Update user profile")
def update_profile(action_context: ActionContext, 
                  username: str, 
                  _auth_token: str) -> dict:
    """Update a user's profile information."""
    # This tool needs auth_token from context
    return make_authenticated_request(_auth_token, username)
The naive solution would be to modify our agent to pass the ActionContext to every tool. However, this would not only clutter our agent’s orchestration logic with dependency management details but also force unnecessary dependencies on tools that don’t need them. Every time the agent calls a tool, it would need to:

Check if the tool needs the ActionContext
Add it to the arguments if needed
Check for any other special dependencies the tool requires
Ensure these don’t conflict with the actual parameters the tool expects
This quickly becomes messy. Here’s what it might look like:

def handle_agent_response(self, action_context: ActionContext, response: str) -> dict:
    """Handle action with dependency injection in the agent."""
    action_def, action = self.get_action(response)
    
    # Agent has to manage all this dependency logic
    args = action["args"].copy()
    if needs_action_context(action_def):
        args["action_context"] = action_context
    if needs_auth_token(action_def):
        args["_auth_token"] = action_context.get("auth_token")
    if needs_user_config(action_def):
        args["_user_config"] = action_context.get("user_config")
        
    result = action_def.execute(**args)
    return result
This is exactly the kind of complexity we want to keep out of our agent. The agent should focus on deciding what actions to take, not on managing how dependencies get passed to tools. Additionally, this approach would make it harder to maintain security and separation of concerns, as every tool would potentially have access to all dependencies.

Instead, we can implement this logic in our environment system, which can examine each tool’s requirements and provide only the dependencies it specifically requests. Consider how much cleaner the agent’s code becomes:

def handle_agent_response(self, action_context: ActionContext, response: str) -> dict:
    """Handle action without dependency management."""
    action_def, action = self.get_action(response)
    result = self.environment.execute_action(self, action_context, action_def, action["args"])
    return result
The agent simply passes everything to the environment and lets it handle the details. The environment can then analyze each tool’s signature and provide exactly the dependencies it needs - no more, no less.

Updating the Environment to Provide Dependencies

Our environment system makes this possible. Let’s walk through how it works.

First, we implement an environment that handles dependency injection:

class PythonEnvironment(Environment):
    def execute_action(self, agent, action_context: ActionContext, 
                      action: Action, args: dict) -> dict:
        """Execute an action with automatic dependency injection."""
        try:
            # Create a copy of args to avoid modifying the original
            args_copy = args.copy()

            # If the function wants action_context, provide it
            if has_named_parameter(action.function, "action_context"):
                args_copy["action_context"] = action_context

            # Inject properties from action_context that match _prefixed parameters
            for key, value in action_context.properties.items():
                param_name = "_" + key
                if has_named_parameter(action.function, param_name):
                    args_copy[param_name] = value

            # Execute the function with injected dependencies
            result = action.execute(**args_copy)
            return self.format_result(result)
        except Exception as e:
            return {
                "tool_executed": False,
                "error": str(e)
            }
The environment examines each tool’s function signature and automatically injects the dependencies it needs. This happens through two mechanisms:

Special parameter names like action_context are automatically injected
Properties from the action_context can be accessed by prefixing the parameter name with _
Excluding Dependencies from the Tool Parameters Schema
Let’s imagine a hypothetical tool for querying a database. The tool might want to access a database connection, configuration settings, and other dependencies like this:

@register_tool()
def query_database(action_context: ActionContext, 
                query: str, 
                _db_connection: DatabaseConnection, 
                _config: dict) -> dict:
    """Process data using external dependencies."""
    # Tool automatically receives db_connection and config
    ... use the database connection ...
    return query_results
We want this tool to automatically receive the dependencies it needs, but we don’t want the agent to have to understand or provide these parameters. The agent should only need to provide the ‘query’ parameter:

# Agent only knows about and provides the data parameter
action = {
    "tool": "query_database",
    "args": {
        "query": "some SQL query"
    }
}
To hide the dependencies from teh agent, we need to update our tool registration system to ignore these special parameters when building the schema that the agent uses:

def get_tool_metadata(func, tool_name=None, description=None, 
                     parameters_override=None, terminal=False, 
                     tags=None):
    """Extract metadata while ignoring special parameters."""
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    args_schema = {
        "type": "object",
        "properties": {},
        "required": []
    }

    for param_name, param in signature.parameters.items():
        # Skip special parameters - agent doesn't need to know about these
        if param_name in ["action_context", "action_agent"] or \
           param_name.startswith("_"):
            continue

        # Add regular parameters to the schema
        param_type = type_hints.get(param_name, str)
        args_schema["properties"][param_name] = {
            "type": "string"  # Simplified for example
        }

        if param.default == param.empty:
            args_schema["required"].append(param_name)

    return {
        "name": tool_name or func.__name__,
        "description": description or func.__doc__,
        "parameters": args_schema,
        "tags": tags or [],
        "terminal": terminal,
        "function": func
    }
Now we can create tools that use rich dependencies while keeping them hidden from the agent. For example, a tool that needs user authentication and configuration:

@register_tool(description="Update user settings in the system")
def update_settings(action_context: ActionContext, 
                   setting_name: str,
                   new_value: str,
                   _auth_token: str,
                   _user_config: dict) -> dict:
    """Update a user setting in the external system."""
    # Tool automatically receives auth_token and user_config
    headers = {"Authorization": f"Bearer {_auth_token}"}
    
    if setting_name not in _user_config["allowed_settings"]:
        raise ValueError(f"Setting {setting_name} not allowed")
        
    response = requests.post(
        "https://api.example.com/settings",
        headers=headers,
        json={"setting": setting_name, "value": new_value}
    )
    
    return {"updated": True, "setting": setting_name}
The agent only sees the setting_name and new_value parameters. When it calls the tool:

# Agent's view of the tool
action = {
    "tool": "update_settings",
    "args": {
        "setting_name": "theme",
        "new_value": "dark"
    }
}
The environment automatically injects the action_context, _auth_token, and _user_config dependencies. This keeps our agent’s orchestration logic clean while providing tools with the rich context they need to function.

This system gives us a clean separation of concerns:

The agent focuses on deciding what actions to take
Tools declare what dependencies they need
The environment handles dependency injection and result management
ActionContext provides a flexible container for shared resources


AI Agents and Agentic AI Architecture in Python
Module 5
Improving AI Agent Reasoning with In-Context Learning

AI Agents and Agentic AI Architecture in Python
Module 5
The Capability Architectural Pattern
Extending the Agent Loop with Capabilities
While tools provide specific functions our agent can use, sometimes we need to extend the agent’s core behavior in more fundamental ways. The Capability pattern allows us to modify multiple aspects of the agent loop while keeping the core logic clean and maintainable.

The idea behind the Capability pattern is to encapsulate specific adaptations of the agent loop inside of a class. This class can be plugged in to modify the behavior of the agent loop without modifying the loop code itself. Agent’s that need more specialized agent loop behavior can be composed by adding capabilities to the agent. The Capability has a lifecycle that begins when the agent loop is about to start and ends when the agent loop is about to terminate. A Capability might open a database connection, log prompts being sent to the LLM, or add metadata to the agent’s responses.

Let’s explore this pattern by implementing something seemingly simple but powerful: making our agent aware of time. An agent that understands time can make better decisions about scheduling, deadlines, and time-sensitive tasks.

The Capability Pattern
A Capability can interact with the agent loop at multiple points. Looking at our Agent class, we can see these interaction points:

def run(self, user_input: str, memory=None, action_context_props=None):

    ... existing code ...
    
    # Initialize capabilities
    for capability in self.capabilities:
        capability.init(self, action_context)
        
    while True:
        # Start of loop capabilities
        can_start_loop = reduce(lambda a, c: c.start_agent_loop(self, action_context),
                              self.capabilities, False)

        ... existing code ...
        
        # Construct prompt with capability modifications
        prompt = reduce(lambda p, c: c.process_prompt(self, action_context, p),
                      self.capabilities, base_prompt)

        ... existing code ...
        
        # Process response with capabilities
        response = reduce(lambda r, c: c.process_response(self, action_context, r),
                        self.capabilities, response)

        ... existing code ...
        
        # Process action with capabilities
        action = reduce(lambda a, c: c.process_action(self, action_context, a),
                      self.capabilities, action)
        
        ... existing code ...
        
        # Process result with capabilities
        result = reduce(lambda r, c: c.process_result(self, action_context, response,
                                                     action_def, action, r),
                       self.capabilities, result)

        ... existing code ...
        
        # End of loop capabilities
        for capability in self.capabilities:
            capability.end_agent_loop(self, action_context)
Each of these interaction points allows a Capability to modify or enhance the agent’s behavior. Let’s implement time awareness using this pattern.

Understanding the Capability Class

A Capability can interact with the agent loop at multiple points. Think of these interaction points like hooks or lifecycle events in a web framework - they give us specific moments where we can modify or enhance the agent’s behavior. Let’s examine the Capability class in detail:

class Capability:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def init(self, agent, action_context: ActionContext) -> dict:
        """Called once when the agent starts running."""
        pass

    def start_agent_loop(self, agent, action_context: ActionContext) -> bool:
        """Called at the start of each iteration through the agent loop."""
        return True

    def process_prompt(self, agent, action_context: ActionContext, 
                      prompt: Prompt) -> Prompt:
        """Called right before the prompt is sent to the LLM."""
        return prompt

    def process_response(self, agent, action_context: ActionContext, 
                        response: str) -> str:
        """Called after getting a response from the LLM."""
        return response

    def process_action(self, agent, action_context: ActionContext, 
                      action: dict) -> dict:
        """Called after parsing the response into an action."""
        return action

    def process_result(self, agent, action_context: ActionContext,
                      response: str, action_def: Action,
                      action: dict, result: any) -> any:
        """Called after executing the action."""
        return result

    def process_new_memories(self, agent, action_context: ActionContext,
                           memory: Memory, response, result,
                           memories: List[dict]) -> List[dict]:
        """Called when new memories are being added."""
        return memories

    def end_agent_loop(self, agent, action_context: ActionContext):
        """Called at the end of each iteration through the agent loop."""
        pass

    def should_terminate(self, agent, action_context: ActionContext,
                        response: str) -> bool:
        """Called to check if the agent should stop running."""
        return False

    def terminate(self, agent, action_context: ActionContext) -> dict:
        """Called when the agent is shutting down."""
        pass
Let’s walk through how these methods map to the agent’s execution cycle:

Initialization Phase The init() method runs once when the agent starts. This is where you set up any initial state or add starting information to the agent’s memory. In our TimeAwareCapability, this is where we first tell the agent what time it is.

Loop Start Phase Before each iteration of the agent loop, start_agent_loop() runs. You can use this to check conditions or prepare for the next iteration. For example, you might want to check if enough time has passed since the last iteration.

Prompt Construction Phase Just before sending a prompt to the LLM, process_prompt() lets you modify the prompt. Our TimeAwareCapability uses this to add current time information to every prompt.

Response Processing Phase After getting the LLM’s response but before parsing it, process_response() lets you modify or validate the raw response text.

Action Processing Phase Once the response is parsed into an action, process_action() lets you modify the action before it’s executed. You might add metadata or validate the action.

Result Processing Phase After the action executes, process_result() lets you modify the result. This is useful for adding additional context or transforming the result format.

Memory Update Phase When new memories are being created, process_new_memories() lets you modify what gets stored in memory. You might add additional context or filter certain types of memories.

Loop End Phase At the end of each iteration, end_agent_loop() runs. This is useful for cleanup or logging what happened during the iteration.

Termination Phase The should_terminate() method can signal that the agent should stop, and terminate() handles any final cleanup when the agent stops.

Each of these methods receives both the agent instance and the ActionContext, giving you access to everything you need to modify the agent’s behavior. The agent processes these methods in sequence using Python’s reduce() function:

# Example from the agent loop
prompt = reduce(lambda p, c: c.process_prompt(self, action_context, p),
               self.capabilities, base_prompt)
Each of these methods receives both the agent instance and the ActionContext, giving you access to everything you need to modify the agent’s behavior. Looking at the Agent constructor, we can see how capabilities become part of the agent:

class Agent:
    def __init__(self,
                 goals: List[Goal],
                 agent_language: AgentLanguage,
                 action_registry: ActionRegistry,
                 generate_response: Callable[[Prompt], str],
                 environment: Environment,
                 capabilities: List[Capability] = [],
                 max_iterations: int = 10,
                 max_duration_seconds: int = 180):
        """
        Initialize an agent with its core GAME components and capabilities.
        
        Goals, Actions, Memory, and Environment (GAME) form the core of the agent,
        while capabilities provide ways to extend and modify the agent's behavior.
        
        Args:
            goals: What the agent aims to achieve
            agent_language: How the agent formats and parses LLM interactions
            action_registry: Available tools the agent can use
            generate_response: Function to call the LLM
            environment: Manages tool execution and results
            capabilities: List of capabilities that extend agent behavior
            max_iterations: Maximum number of action loops
            max_duration_seconds: Maximum runtime in seconds
        """
        self.goals = goals
        self.generate_response = generate_response
        self.agent_language = agent_language
        self.actions = action_registry
        self.environment = environment
        self.capabilities = capabilities or []
        self.max_iterations = max_iterations
        self.max_duration_seconds = max_duration_seconds
This design lets us compose an agent with exactly the capabilities it needs. For example, we might create an agent that’s both time-aware and able to log its actions:

agent = Agent(
    goals=[
        Goal(name="scheduling",
             description="Schedule meetings considering current time and availability")
    ],
    agent_language=JSONAgentLanguage(),
    action_registry=registry,
    generate_response=llm.generate,
    environment=PythonEnvironment(),
    capabilities=[
        TimeAwareCapability(),
        LoggingCapability(log_level="INFO"),
        MetricsCapability(metrics_server="prometheus:9090")
    ]
)
Each capability in the list gets a chance to participate in every phase of the agent’s execution. The agent processes these methods in sequence using Python’s reduce() function:

The TimeAwareCapability might add time information to a prompt, and then the LoggingCapability could log that time-enhanced prompt before it goes to the LLM.

This architecture allows us to build complex behaviors by composing simple, focused capabilities, each responsible for one aspect of the agent’s behavior. It’s similar to how middleware works in web frameworks, where each piece can modify the request/response cycle without the core application needing to know about these modifications.

Implementing Time Awareness
The TimeAwareCapability needs to inform the agent about the current time and ensure this information persists throughout its decision-making process:

from datetime import datetime
from zoneinfo import ZoneInfo

class TimeAwareCapability(Capability):
    def __init__(self):
        super().__init__(
            name="Time Awareness",
            description="Allows the agent to be aware of time"
        )
        
    def init(self, agent, action_context: ActionContext) -> dict:
        """Set up time awareness at the start of agent execution."""
        # Get timezone from context or use default
        time_zone_name = action_context.get("time_zone", "America/Chicago")
        timezone = ZoneInfo(time_zone_name)
        
        # Get current time in specified timezone
        current_time = datetime.now(timezone)
        
        # Format time in both machine and human-readable formats
        iso_time = current_time.strftime("%Y-%m-%dT%H:%M:%S%z")
        human_time = current_time.strftime("%H:%M %A, %B %d, %Y")
        
        # Store time information in memory
        memory = action_context.get_memory()
        memory.add_memory({
            "type": "system",
            "content": f"""Right now, it is {human_time} (ISO: {iso_time}).
            You are in the {time_zone_name} timezone.
            Please consider the day/time, if relevant, when responding."""
        })
        
    def process_prompt(self, agent, action_context: ActionContext, 
                      prompt: Prompt) -> Prompt:
        """Update time information in each prompt."""
        time_zone_name = action_context.get("time_zone", "America/Chicago")
        current_time = datetime.now(ZoneInfo(time_zone_name))
        
        # Add current time to system message
        system_msg = (f"Current time: "
                     f"{current_time.strftime('%H:%M %A, %B %d, %Y')} "
                     f"({time_zone_name})\n\n")
        
        # Add to existing system message or create new one
        messages = prompt.messages
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = system_msg + messages[0]["content"]
        else:
            messages.insert(0, {
                "role": "system",
                "content": system_msg
            })
            
        return Prompt(messages=messages)
Now we can use this capability when creating our agent:

agent = Agent(
    goals=[Goal(name="task", description="Complete the assigned task")],
    agent_language=JSONAgentLanguage(),
    action_registry=registry,
    generate_response=llm.generate,
    environment=PythonEnvironment(),
    capabilities=[
        TimeAwareCapability()
    ]
)
Our agent now consistently knows the current time, enabling it to make time-aware decisions. For example, if we ask it to schedule a meeting, it might respond:

# Example conversation
agent.run("Schedule a team meeting for today")

# Agent response might include:
"Since it's already 5:30 PM on Friday, I recommend scheduling the meeting 
for Monday morning instead. Would you like me to look for available times 
on Monday?"

How Time Awareness Changes Agent Behavior
The TimeAwareCapability modifies agent behavior in several ways:

Through init(): When the agent starts, it establishes baseline time awareness by adding time information to memory.

Through process_prompt(): Before each prompt, it updates the current time, ensuring the agent always has fresh time data for decision-making.

The capability’s modifications ripple through the agent’s decision-making process while keeping the core agent loop clean. We didn’t need to modify the Agent class at all - the capability pattern handled everything.

Extending the Time Awareness Capability
We could extend this capability further to handle more complex time-related features:

class EnhancedTimeAwareCapability(TimeAwareCapability):
    def process_action(self, agent, action_context: ActionContext, 
                      action: dict) -> dict:
        """Add timing information to action results."""
        # Add execution time to action metadata
        action["execution_time"] = datetime.now(
            ZoneInfo(action_context.get("time_zone", "America/Chicago"))
        ).isoformat()
        return action
        
    def process_result(self, agent, action_context: ActionContext,
                      response: str, action_def: Action,
                      action: dict, result: any) -> any:
        """Add duration information to results."""
        if isinstance(result, dict):
            result["action_duration"] = (
                datetime.now(ZoneInfo(action_context.get("time_zone"))) -
                datetime.fromisoformat(action["execution_time"])
            ).total_seconds()
        return result
This enhanced version tracks when actions are executed and how long they take, building a richer understanding of time in the agent’s operation.


AI Agents and Agentic AI Architecture in Python
Module 5
Ahead of Time Planning for Improving Agent Reasoning

Plan First: A New Capability

One key to making agents more effective is getting them to think strategically before taking action. Instead of jumping straight into executing tools, we want our agent to first develop a comprehensive plan. Let’s build a capability that enforces this “plan first” approach.

The Plan First Pattern
Here’s how we’ll make our agent plan before acting:

When the agent first starts, we’ll prompt it to create a detailed plan
We’ll store this plan in the agent’s memory
The agent will refer back to this plan throughout its execution
Let’s implement this using a capability:

class PlanFirstCapability(Capability):
    def __init__(self, plan_memory_type="system", track_progress=False):
        super().__init__(
            name="Plan First Capability",
            description="The Agent will always create a plan and add it to memory"
        )
        self.plan_memory_type = plan_memory_type
        self.first_call = True
        self.track_progress = track_progress

    def init(self, agent, action_context):
        if self.first_call:
            self.first_call = False
            plan = create_plan(
                action_context=action_context,
                memory=action_context.get_memory(),
                action_registry=action_context.get_action_registry()
            )

            action_context.get_memory().add_memory({
                "type": self.plan_memory_type,
                "content": "You must follow these instructions carefully to complete the task:\n" + plan
            })
The key is in our create_plan function, which guides the LLM through structured thinking:

@register_tool(tags=["planning"])
def create_plan(action_context: ActionContext,
                memory: Memory,
                action_registry: ActionRegistry) -> str:
   """Create a detailed execution plan based on the task and available tools."""

   # Get tool descriptions for the prompt
   tool_descriptions = "\n".join(
      f"- {action.name}: {action.description}"
      for action in action_registry.get_actions()
   )

   # Get relevant memory content
   memory_content = "\n".join(
      f"{m['type']}: {m['content']}"
      for m in _memory.items
      if m['type'] in ['user', 'system']
   )

   # Construct the prompt as a string
   prompt = f"""Given the task in memory and the available tools, create a detailed plan.
Think through this step by step:

1. First, identify the key components of the task
2. Consider what tools you have available
3. Break down the task into logical steps
4. For each step, specify:
   - What needs to be done
   - What tool(s) will be used
   - What information is needed
   - What the expected outcome is

Write your plan in clear, numbered steps. Each step should be specific and actionable.

Available tools:
{tool_descriptions}

Task context from memory:
{memory_content}

Create a plan that accomplishes this task effectively."""

   return prompt_llm(action_context=action_context, prompt=prompt)
The goal is to get the agent to create and include a plan for its work at the start of the conversation. For example, for a data analysis task, the agent might generate a plan like this:

# With planning
agent_response = """Plan for Sales Data Analysis:

1. Data Validation
   - Tool: validate_data()
   - Check data completeness and format
   - Ensure all required fields are present
   - Expected: Confirmation of valid dataset

2. Initial Analysis
   - Tool: analyze_data()
   - Calculate key metrics (revenue, growth)
   - Generate summary statistics
   - Expected: Basic statistical overview

3. Trend Identification
   - Tool: find_patterns()
   - Look for seasonal patterns
   - Identify sales trends
   - Expected: List of significant trends

4. Visualization
   - Tool: create_visualization()
   - Create relevant charts
   - Highlight key findings
   - Expected: Clear visual representations

5. Report Generation
   - Tool: generate_report()
   - Compile findings
   - Include visualizations
   - Expected: Comprehensive report

I'll begin with step 1: Data Validation..."""
This enhanced version:

Creates a detailed plan upfront
Stores the plan in memory
Helps maintain focus on the overall strategy
Usage:

agent = Agent(
    goals=[
        Goal(name="analysis",
             description="Analyze sales data and create a report")
    ],
    capabilities=[
        PlanFirstCapability(track_progress=True)
    ],
    # ... other agent configuration
)

result = agent.run("Analyze our Q4 sales data and create a report")

AI Agents and Agentic AI Architecture in Python
Module 5
Intermediate Planning: Tracking Progress in the Agent Loop

Tracking Progress
In complex tasks, agents need to periodically step back and assess their progress. Let’s build a capability that adds reflection and progress tracking to the end of each agent loop iteration, allowing the agent to understand what it just did and plan its next steps more effectively.

To achieve this, we introduce a track_progress function that evaluates the agent’s current state after each action. This function operates similarly to our planning function but shifts the focus to assessment rather than forward planning. By analyzing available tools, memory context, and completed steps, the agent can identify what has been accomplished, any blockers encountered, and what should happen next. This ensures the agent maintains awareness of its trajectory rather than blindly executing actions without reflection.

By integrating track_progress at the end of each loop iteration, we enable the agent to continuously refine its strategy. Instead of relying solely on a predefined plan, the agent dynamically adapts based on real-time feedback. This aligns with how human problem-solving works—we plan, act, evaluate, and adjust. With this addition, our agent becomes more resilient and capable, recognizing obstacles early and making course corrections as needed, and potentially leading to more efficient and intelligent execution of complex workflows.

@register_tool(tags=["prompts"])
def track_progress(action_context: ActionContext,
                   _memory: Memory,
                   action_registry: ActionRegistry) -> str:
    """Generate a progress report based on the current task, available tools, and memory context."""

    # Get tool descriptions for the prompt
    tool_descriptions = "\n".join(
        f"- {action.name}: {action.description}"
        for action in action_registry.get_actions()
    )

    # Get relevant memory content
    memory_content = "\n".join(
        f"{m['type']}: {m['content']}"
        for m in _memory.items
        if m['type'] in ['user', 'system']
    )

    # Construct the prompt as a string
    prompt = f"""Given the current task and available tools, generate a progress report.
Think through this step by step:

1. Identify the key components of the task and the intended outcome.
2. Assess the progress made so far based on available information.
3. Identify any blockers or issues preventing completion.
4. Suggest the next steps to move forward efficiently.
5. Recommend any tool usage that might help complete the task.

Write your progress report in clear, structured points.

Available tools:
{tool_descriptions}

Task context from memory:
{memory_content}

Provide a well-organized report on the current progress and next steps."""

    return prompt_llm(action_context=action_context, prompt=prompt)
Building a Progress Tracking Capability
Now that we have a track_progress function, let’s turn it into a Capability that we can add to our agent. This capability adds overhead and slows down our agent, since it will add one prompt per agent loop, but it can help with complex tasks where tracking progress is essential. Here’s how we can implement a capability that tracks progress after each action:

class ProgressTrackingCapability(Capability):
    def __init__(self, memory_type="system", track_frequency=1):
        super().__init__(
            name="Progress Tracking",
            description="Tracks progress and enables reflection after actions"
        )
        self.memory_type = memory_type
        self.track_frequency = track_frequency
        self.iteration_count = 0

    def end_agent_loop(self, agent, action_context: ActionContext):
        """Generate and store progress report at the end of each iteration."""
        self.iteration_count += 1
        
        # Only track progress on specified iterations
        if self.iteration_count % self.track_frequency != 0:
            return
            
        # Get the memory and action registry from context
        memory = action_context.get_memory()
        action_registry = action_context.get_action_registry()
        
        # Generate progress report
        progress_report = track_progress(
            action_context=action_context,
            _memory=memory,
            action_registry=action_registry
        )
        
        # Add the progress report to memory
        memory.add_memory({
            "type": self.memory_type,
            "content": f"Progress Report (Iteration {self.iteration_count}):\n{progress_report}"
        })

    This capability uses the track_progress tool to generate detailed progress reports. Let’s see how it transforms agent behavior in practice:

# Create an agent with progress tracking
agent = Agent(
    goals=[
        Goal(
            name="data_processing",
            description="Process and analyze customer feedback data"
        )
    ],
    capabilities=[
        ProgressTrackingCapability(track_frequency=2)  # Track every 2nd iteration
    ],
    # ... other agent configuration
)

# Example execution flow
memory = agent.run("Analyze customer feedback from Q4 and identify top issues")
After each iteration (or every N iterations), the agent will pause to reflect.

Benefits of End-of-Loop Progress Tracking
Tracking progress at the end of each loop iteration (rather than the beginning) offers several advantages:

The agent can assess the impact of its most recent action
Memory contains fresh information about what just happened so that the plan is based on the latest data as opposed to outdated assumptions used in the original plan
The agent can adapt its strategy based on actual results
Progress reports create a clear audit trail of agent decision-making
Using Progress Reports in Decision Making
The stored progress reports become part of the agent’s memory, influencing future decisions. When the agent needs to choose its next action, it can reference these reports to:

Avoid repeating completed steps
Address identified blockers
Follow through on recommended next steps
Use suggested tools effectively
The combination of immediate reflection and persistent memory can help the agent maintain focus on its goals while adapting to new information and changing circumstances. This can be valuable for long-running tasks where maintaining context is crucial for success and identifying when things are going wrong and adapting is important.












