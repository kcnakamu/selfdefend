import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class HuggingFaceLLM:
    MODEL_PATHS = {
        "llama-2-7b": "./checkpoint/Llama-2-7b-chat-hf",
        "llama-2-13b": "./checkpoint/Llama-2-13b-chat-hf",
        "mistral-7b-v0.2": "./checkpoint/Mistral-7B-Instruct-v0.2",
        "TinyLlama-1.1B-lora": {
            "base": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "adapter": "./checkpoint/TinyLlama-1.1B-lora-direct-epoch1-bs2-lr0.0002"
        }
    }

    LORA_BASE_MODELS = {
        "TinyLlama-1.1B-lora-direct-epoch1-bs2-lr0.0002": "TinyLlama-1.1B",
    }

    def __init__(self, model_name="llama-2-7b", device="cuda") -> None:
        self.model_name = model_name

        if isinstance(self.MODEL_PATHS[model_name], dict):
            base_path = self.MODEL_PATHS[model_name]["base"]
            adapter_path = self.MODEL_PATHS[model_name]["adapter"]
            
            self.tokenizer = AutoTokenizer.from_pretrained(base_path)
            base_model = AutoModelForCausalLM.from_pretrained(base_path)
            
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            self.model.eval() 
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATHS[model_name])
            if '13b' in model_name:
                self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_PATHS[model_name], torch_dtype=torch.float16, device_map='auto')
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_PATHS[model_name], device_map='auto')
            self.model.to(device)
            self.model.eval()

        self.device = device

    def __call__(self, msg):
        inputs = self.tokenizer(msg, return_tensors='pt').to(self.device)
        with torch.no_grad():
            generate_ids = self.model.generate(inputs['input_ids'].to(self.device), pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=128)
            out = self.tokenizer.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return out
