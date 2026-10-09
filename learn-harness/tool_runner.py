# this just check that the tool call is valid and then runs it all fromthe def in tools.py

from tools import TOOLS, TOOL_NAMES
import tools

#check then runs if valid
def tool_runner(tool_name, tool_parameters):
    # check if the tool call is valid
    is_valid, message = check_tool_call(tool_name, tool_parameters)
    if not is_valid:
        return f"Invalid tool call: {message}"

    # execute import the tool function and run it with the parameters
    try:
        tool_func = getattr(tools, tool_name) #import the function from tools.py
        res = tool_func(**tool_parameters) # unpack the parameters and pass them to the function
        return res
    except KeyboardInterrupt: # Ctrl-C while a tool runs stops the tool, not the whole harness
        return "Tool execution was interrupted by the user."
    except Exception as e:
        return f"An error occurred while executing the tool: {e}"


#cheks if tool exists and if the parameters are valid
def check_tool_call(tool_name, tool_parameters):
    # check if the tool name is in the TOOLS dictionary
    if tool_name not in TOOL_NAMES:
        return False, f"Tool '{tool_name}' is not recognized."

    # check that the arguments are a dict, otherwise **tool_parameters makes no sense
    if not isinstance(tool_parameters, dict):
        return False, f"Arguments for tool '{tool_name}' must be a JSON object."

    # the schema lives under TOOLS[i]["function"]["parameters"]
    tool_schema = TOOLS[TOOL_NAMES.index(tool_name)]["function"]["parameters"]

    # check if the required parameters are present
    required_params = tool_schema.get("required", [])
    for param in required_params:
        if param not in tool_parameters:
            print(f"the params present are: {tool_parameters}")
            return False, f"Missing required parameter: '{param}' for tool '{tool_name}'."

    # check that there are no parameters the tool doesn't define
    allowed_params = tool_schema.get("properties", {})
    for param in tool_parameters:
        if param not in allowed_params:
            return False, f"Unknown parameter: '{param}' for tool '{tool_name}'."

    return True, "Tool call is valid."