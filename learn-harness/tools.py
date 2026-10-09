import subprocess
import glob

def run_command(command):
    """
    Executes a shell command and returns the output.
    
    Args:
        command (str): The shell command to execute.
    Returns:
        str: The output of the command.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"An error occurred while executing the command: {e.stderr}"


def run_python_code(code):
    """
    Executes the given Python code and returns the result.
    
    Args:
        code (str): The Python code to execute.
    Returns:
        str: The result of the executed code.
    """
    try:
        result = subprocess.run(["python", "-c", code], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"An error occurred while executing the Python code: {e.stderr}"

# def get_time():
#     """
#     Returns the current system time.
    
#     Returns:
#         str: The current system time.
#     """
#     from datetime import datetime
#     return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# def ls(path):
#     """
#     Lists the contents of a directory.
    
#     Args:
#         path (str): The directory path. Defaults to the current directory.
#     Returns:
#         str: The output of the 'ls' command.
#     """
#     try:
#         result = subprocess.run(["ls", path], capture_output=True, text=True, check=True)
#         return result.stdout
#     except subprocess.CalledProcessError as e:
#         return f"An error occurred while listing the directory: {e.stderr}"

# def read_file(file_path, start_line=None, end_line=None):
#     """
#     Reads the contents of a file.
    
#     Args:
#         file_path (str): The path to the file.
#         start_line (int, optional): The line to start reading from. Defaults to None.
#         end_line (int, optional): The line to stop reading at. Defaults to None.
#     Returns:
#         str: The contents of the file.
#     """
#     try:
#         with open(file_path, 'r') as file:
#             lines = file.readlines()
#             if start_line is not None:
#                 lines = lines[start_line:]
#             if end_line is not None:
#                 lines = lines[:end_line]

#             res = ''.join(lines)
#             if len(res) > 10_000:
#                 return res[:10000] + "\n\n[Output truncated to 10_000 characters. The full output is too long to display.]"
#             return ''.join(lines)
#     except Exception as e:
#         return f"An error occurred while reading the file: {e}"

# def write_file(file_path, content):
#     """
#     Writes content to a file.
    
#     Args:
#         file_path (str): The path to the file.
#         content (str): The content to write to the file.
#     Returns:
#         str: A message indicating success or failure.
#     """
#     try:
#         with open(file_path, 'w') as file:
#             file.write(content)
#         return f"Successfully wrote to {file_path}."
#     except Exception as e:
#         return f"An error occurred while writing to the file: {e}"

# def replace_file_content(file_path, old_text, new_text):
#     """
#     Replaces occurrences of old_text with new_text in a file.
    
#     Args:
#         file_path (str): The path to the file.
#         old_text (str): The text to be replaced.
#         new_text (str): The text to replace with.
#     Returns:
#         str: A message indicating success or failure.
#     """
#     try:
#         with open(file_path, 'r') as file:
#             content = file.read()
        
#         content = content.replace(old_text, new_text)
        
#         with open(file_path, 'w') as file:
#             file.write(content)
        
#         return f"Successfully replaced '{old_text}' with '{new_text}' in {file_path}."
#     except Exception as e:
#         return f"An error occurred while replacing text in the file: {e}"

# def edit_file(file_path, new_content, start_line, end_line):
#     """
#     Edits a file by replacing its content with new_content at the specified range.
#     Args:
#         file_path (str): The path to the file.
#         new_content (str): The new content to write to the file.\
#         start_line (int): The line number to start replacing content from.
#         end_line (int): The line number to stop replacing content at.
#     Returns:
#         str: A message indicating success or failure.
#     """
#     try:
#         with open(file_path, 'r') as file:
#             lines = file.readlines()
        
#         if start_line < 0 or start_line >= len(lines):
#             return f"Error: start_line {start_line} is out of range for the file with {len(lines)} lines."
        
#         if end_line < start_line or end_line > len(lines):
#             return f"Error: end_line {end_line} is out of range for the file with {len(lines)} lines."
        
#         lines[start_line:end_line] = [new_content + '\n']
        
#         with open(file_path, 'w') as file:
#             file.writelines(lines)
        
#         return f"Successfully edited {file_path} starting from line {start_line}."
#     except Exception as e:
#         return f"An error occurred while editing the file: {e}"

# def glob(pattern):
#     """
#     Returns a list of file paths matching the given pattern.
    
#     Args:
#         pattern (str): The glob pattern to match files.
#     Returns:
#         list: A list of matching file paths.
#     """
#     return glob.glob(pattern)

# def grep(pattern, file_path):
#     """
#     Searches for a pattern in a file and returns the matching lines.
    
#     Args:
#         pattern (str): The regex pattern to search for.
#         file_path (str): The path to the file.
#     Returns:
#         list: A list of matching lines.
#     """
#     import re
#     try:
#         with open(file_path, 'r') as file:
#             lines = file.readlines()
        
#         matching_lines = [line for line in lines if re.search(pattern, line)]
        
#         return matching_lines
#     except Exception as e:
#         return f"An error occurred while searching the file: {e}"

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
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "get_time",
    #         "description": "Returns the current time.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #             },
    #             "required": []
    #         },
    #     },
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "read_file",
    #         "description": "Reads the contents of a file.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "file_path": {
    #                     "type": "string",
    #                     "description": "The path to the file."
    #                 },
    #                 "start_line": {
    #                     "type": ["number", "integer"],
    #                     "description": "The line to start reading from. Defaults to None."
    #                 },
    #                 "end_line": {
    #                     "type": ["number", "integer"],
    #                     "description": "The line to stop reading at. Defaults to None."
    #                 }
    #             },
    #             "required": ["file_path"]
    #         },
    #     },
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "write_file",
    #         "description": "Writes content to a file.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "file_path": {
    #                     "type": "string",
    #                     "description": "The path to the file."
    #                 },
    #                 "content": {
    #                     "type": "string",
    #                     "description": "The content to write to the file."
    #                 }
    #             },
    #             "required": ["file_path", "content"]
    #         },
    #     },
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "replace_file_content",
    #         "description": "Replaces occurrences of old_text with new_text in a file.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "file_path": {
    #                     "type": "string",
    #                     "description": "The path to the file."
    #                 },
    #                 "old_text": {
    #                     "type": "string",
    #                     "description": "The text to be replaced."
    #                 },
    #                 "new_text": {
    #                     "type": "string",
    #                     "description": "The text to replace with."
    #                 }
    #             },
    #             "required": ["file_path", "old_text", "new_text"]
    #         },
    #     },
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "edit_file",
    #         "description": "Edits a file by replacing its content with new_content at the specified range.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "file_path": {
    #                     "type": "string",
    #                     "description": "The path to the file."
    #                 },
    #                 "new_content": {
    #                     "type": "string",
    #                     "description": "The new content to write to the file."
    #                 },
    #                 "start_line": {
    #                     "type": ["number", "integer"],
    #                     "description": "The line number to start replacing content from."
    #                 },
    #                 "end_line": {
    #                     "type": ["number", "integer"],
    #                     "description": "The line number to stop replacing content at."
    #                 }
    #             },
    #             "required": ["file_path", "new_content", "start_line", "end_line"]
    #         },
    #     },
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "glob",
    #         "description": "Returns a list of file paths matching the given pattern.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "pattern": {
    #                     "type": "string",
    #                     "description": "The glob pattern to match files."
    #                 }
    #             },
    #             "required": ["pattern"]
    #         },
    #     },
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "grep",
    #         "description": "Searches for a pattern in a file and returns the matching lines.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "pattern": {
    #                     "type": "string",
    #                     "description": "The regex pattern to search for."
    #                 },
    #                 "file_path": {
    #                     "type": "string",
    #                     "description": "The path to the file."
    #                 }
    #             },
    #             "required": ["pattern", "file_path"]
    #         },
    #     },
    # }
]

TOOL_NAMES = [tool["function"]["name"] for tool in TOOLS]
