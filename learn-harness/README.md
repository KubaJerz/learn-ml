## This is a very light weight and simpel llm harnessbuilt form scratch by had to better undersatn harnesses. 


Fundamentally, the model still just spits out text, but there are a few things:
1. To even get it to chat like a chatbot, you must give it the template it was trained on. 
    - For Qwen (ChatML):
        ```
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        hi<|im_end|>
        <|im_start|>assistant
        ```
        The last line is left open so the model continues from there (`add_generation_prompt=True`).
2. It'll just act like a chatbot. If it was also fine-tuned on agentic tasks, then you can give it tool defs, which just need to be in the input prompt, and each model will print those out in a different way.
    - For Qwen: `"<tool_call>(.*?)</tool_call>"` blank, blank.
3. You simply just parse the script for tool calls, run them, and return the result to the model. There's a loop for that until the model replies without a tool call (the model decides when it's done, not the harness).
4. The output has no tool calls, and then you just print that out to the user.

- Every model has its own chat format, its own way of showing tool definitions, and its own tool-call output format.
- The librarys handle:
    - Chat templet
    - Tool defs
- The harness needs to handle:
    - parsingthe output format from each model type 

## Harness.py is the continuous agent to loop. The other files either define tools or instantiate the model. 