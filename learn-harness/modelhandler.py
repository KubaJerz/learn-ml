from transformers import AutoModelForCausalLM, AutoTokenizer

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
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device) #tokenize the formatted string and return at tensor
        outputs = self.model.generate(**inputs, max_new_tokens=2048) # forawrd pass though
        return self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]-1:], skip_special_tokens=True) #decode jus the new tokens


# def whole(device=device):
#     model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.8-27B", dtype="auto", device_map=device)
#     tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
#     return model, tokenizer

# 
# processor = AutoProcessor.from_pretrained("Qwen/Qwen3.8-27B")
# model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.8-27B", device_map="auto")
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
#             {"type": "text", "text": "What animal is on the candy?"}
#         ]
#     },
# ]
# inputs = processor.apply_chat_template(
# 	messages,
# 	add_generation_prompt=True,
# 	tokenize=True,
# 	return_dict=True,
# 	return_tensors="pt",
# ).to(model.device)

# outputs = model.generate(**inputs, max_new_tokens=256)
# print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))