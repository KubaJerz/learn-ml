import subprocess
import sys
import glob

#
#
# This code defines the available tools for the model.
# Fundamentally, the only tool you really need is the run command, but many other ones are good quality of life. 
# Particular file editing tools would be helpful for the model, but are not necessary. 

MAX_OUTPUT_CHARS = 10_000 # cap tool output so one big `cat` doesn't blow up the context window


# turns a finished subprocess into one string the model can read: exit code, stdout and stderr together
def format_result(result):
    output = f"exit code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    if len(output) > MAX_OUTPUT_CHARS:
        output = "[output truncated, showing the end]\n" + output[-MAX_OUTPUT_CHARS:]
    return output


def run_command(command):
    """
    Executes a shell command and returns the output.
    
    Args:
        command (str): The shell command to execute.
    Returns:
        str: The exit code, stdout and stderr of the command.
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return format_result(result)


def run_python_code(code):
    """
    Executes the given Python code and returns the result.
    
    Args:
        code (str): The Python code to execute.
    Returns:
        str: The exit code, stdout and stderr of the executed code.
    """
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True) # sys.executable = same python as the harness
    return format_result(result)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Executes a shell command and returns the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute."
                    }
                },
                "required": ["command"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Executes the given Python code and returns the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute."
                    }
                },
                "required": ["code"]
            },
        },
    },
  
]

TOOL_NAMES = [tool["function"]["name"] for tool in TOOLS]
