from transformers import AutoModelForCausalLM, AutoTokenizer

#
#  This code hanndels loading the model and tokenizer
#
#


class ModelHandler:
    # loades the model and tokenizer from the pretrained model and returns the class object 
    def __init__(self, tools=None, device="cuda"):
        self.device = device
        self.TOOLS = tools
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def __call__(self, input_text):
        formatted = self.tokenizer.apply_chat_template(input_text, add_generation_prompt=True, tools=self.TOOLS, tokenize=False) #take list of dicts to "<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n" string format
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device) #tokenize the formatted string and return at tensor
        outputs = self.model.generate(**inputs, max_new_tokens=2048) # forawrd pass though
        return self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True) #decode jus the new tokens
