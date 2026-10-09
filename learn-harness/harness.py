import json
import sys
from modelhandler import ModelHandler
from tools import TOOLS
from tool_runner import tool_runner
import re

#
#
# This is the main script that defines the harness. 
#
# 

print("Welcome !\nModel is loading, please wait ...")
model_handler = ModelHandler(device="cuda", tools=TOOLS)
print("Model is loaded, you can start chatting now !\n")

chat_history = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant running on the user's computer. "
            "You have access to tools, which are listed below.\n\n"
        ),
    }
]



while True:

    ####################################################
    # 1. prepare the user input for processing and check for exit phrases
    ####################################################
    try:
        #prepare the user input for processing
        user_input = input("\nYou: ").strip()

    # check for keyboard interrupt (Ctrl-C) or end of input (Ctrl-D) and exit gracefully
    except (KeyboardInterrupt, EOFError):
        print("\nExiting the chat. Goodbye!")
        print(json.dumps(chat_history, indent=2, ensure_ascii=False)) 
        break

    # don't send empty messages to the model
    if not user_input:
        continue

    #check for exit "phrases"
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break

    chat_history.append({"role": "user", "content": user_input})

    ####################################################
    # 2. pass model handler (aka tokenizer and model)
    ####################################################
    try:
        response = model_handler(chat_history)
        chat_history.append({"role": "assistant", "content": response})
    except KeyboardInterrupt:
        print("\nGeneration interrupted.")
        continue
    except Exception as e:
        print(f"An error occurred while processing the input: {e}")
        continue # don't fall through with a stale (or undefined) response

    ####################################################
    # 3. now harness kicks in a interprerprests the models output 
    ####################################################
    # 3.a check if the model output has a tool in it
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_calls = re.findall(tool_call_pattern, response, flags=re.DOTALL) # the DORTALL alows the '.' to match new lines too
    if not tool_calls: # 3.a no tool calls found, just print the model response
        print(f"\nModel: {response}")
    else:# 3.b response contains tool calls
        has_called_tool = True
        while has_called_tool:
            #3.b.1 first print the model response without the tool calls 
            response_without_tool_calls = re.sub(tool_call_pattern, "", response, flags=re.DOTALL)
            if response_without_tool_calls.strip():  # Check if the response without tool calls is not empty
                print(f"\nModel: {response_without_tool_calls}")

            #3.b.2 then iterate over the tool calls and turn each one into raw json
            for tool_call in tool_calls:
                try:
                    tool_data = json.loads(tool_call)
                except json.JSONDecodeError as e:
                    # models produce broken json sometimes, tell the model so it can retry instead of crashing
                    chat_history.append({"role": "tool", "content": f"Could not parse tool call as JSON: {e}"})
                    continue
                if not isinstance(tool_data, dict):
                    chat_history.append({"role": "tool", "content": "Tool call must be a JSON object with 'name' and 'arguments'."})
                    continue

                #3.b.3 extract the tool name and parameters
                tool_name = tool_data.get("name", "Unknown")
                tool_parameters = tool_data.get("arguments", {})

                #3.b.4 then ask the human before running anything, the model never runs a tool on its own
                # this is the main safety check: tool output goes back into the model, so a file or web page
                # it reads can contain instructions (prompt injection). The human reading each call is the defence.
                print(f"\nModel wants to run '{tool_name}' with:\n{json.dumps(tool_parameters, indent=2, ensure_ascii=False)}")
                try:
                    answer = input("Allow? [y/N] ").strip().lower()
                except (KeyboardInterrupt, EOFError): # Ctrl-C or Ctrl-D counts as a no
                    answer = ""
                is_valid = answer == "y"

                #3.b.5 if the tool call is allowed, execute the tool and get the result
                if is_valid:
                    tool_result = tool_runner(tool_name, tool_parameters)
                else:
                    tool_result = "The user denied this tool call."
                chat_history.append({"role": "tool", "content": tool_result})

            #3.b.6 pass the tool results back to the model for further processing
            try:
                response = model_handler(chat_history)
                chat_history.append({"role": "assistant", "content": response})
                if not re.search(tool_call_pattern, response, flags=re.DOTALL):
                    has_called_tool = False
                    print(f"\nModel Response: {response}")
                else:
                    tool_calls = re.findall(tool_call_pattern, response, flags=re.DOTALL) # the DORTALL alows the '.' to match new lines too
            except KeyboardInterrupt:
                print("\nGeneration interrupted.")
                break
            except Exception as e:
                print(f"An error occurred while processing the input after tool execution: {e}")
                break # stop here, otherwise the loop re-runs the same tool calls forever
            

