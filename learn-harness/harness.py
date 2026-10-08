import json
import sys
from modelhandler import ModelHandler
from tools import TOOLS
from tool_runner import tool_runner
import re

print("Welcome !\nModel is loading, please wait ...")
model_handler = ModelHandler(device="cuda", tools=TOOLS)
print("Model is loaded, you can start chatting now !\n")
chat_history = []
while True:

    ####################################################
    # 1. prepare the user input for processing and check for exit phrases
    ####################################################
    try:
        #prepare the user input for processing
        user_input = sys.stdin.readline().strip()

        #check for exit "phrases"
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chat. Goodbye!")
            break


    # check for keyboard interrupt and exit gracefully
    except KeyboardInterrupt:
        print("\nExiting the chat. Goodbye!")
        print(f"Chat history: {chat_history}")
        break

    # check for any other exceptions and print the error message
    except Exception as e:
        print(f"An error occurred: {e}")

    chat_history.append({"role": "user", "content": user_input})

    ####################################################
    # 2. pass model handler (aka tokenizer and model)
    ####################################################
    try:
        response = model_handler(chat_history)
        chat_history.append({"role": "assistant", "content": response})
    except Exception as e:
        print(f"An error occurred while processing the input: {e}")

    ####################################################
    # 3. now harness kicks in a interprerprests the models output 
    ####################################################
    # 3.a check if the model output has a tool in it
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_calls = re.findall(tool_call_pattern, response, flags=re.DOTALL) # the DORTALL alows the '.' to match new lines too
    print(f"Tool calls found: {tool_calls}")
    if not tool_calls: # 3.a no tool calls found, just print the model response
        print(f"\n\nModel: {response}\n\n")
    else:# 3.b response contains tool calls
        has_called_tool = True
        while has_called_tool:
            #3.b.1 first print the model response without the tool calls 

            #temp   
            print(f"TEMP TEMP model response: {response}")


            response_without_tool_calls = re.sub(tool_call_pattern, "", response, flags=re.DOTALL)
            if response_without_tool_calls.strip():  # Check if the response without tool calls is not empty
                print(f"\n\nModel: {response_without_tool_calls}\n\n")

            #3.b.2 then turn tool calls in raw json
            raw_json = [json.loads(tool_call) for tool_call in tool_calls]

            #3.b.3 then iterate over the raw json and extract the tool name and parameters
            for tool_data in raw_json:
                tool_name = tool_data.get("name", "Unknown")
                tool_parameters = tool_data.get("arguments", {})

                #3.b.4 then check that the tool call is allowed with the params
                is_valid = True

                #3.b.5 if the tool call is allowed, execute the tool and get the result
                if is_valid:
                    tool_result = tool_runner(tool_name, tool_parameters)
                    chat_history.append({"role": "tool", "content": tool_result})

            #3.b.6 pass the tool results back to the model for further processing
            try:
                response = model_handler(chat_history)
                chat_history.append({"role": "assistant", "content": response})
                if not re.search(tool_call_pattern, response, flags=re.DOTALL):
                    has_called_tool = False
                    print(f"\n\nModel Response: {response}\n\n")
            except Exception as e:
                print(f"An error occurred while processing the input after tool execution: {e}")
            

