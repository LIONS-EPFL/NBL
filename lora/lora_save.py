from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
import torch
import pickle

# === Load model & tokenizer ===
model_path = "/NBL/llama_model_ds"  # Replace with your local model path
base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
# llama_model_ds2

# Load model & tokenizer
#model_id = "/NBL/ds_nbl_16" #"meta-llama/Meta-Llama-3-8B-Instruct"
#tokenizer = AutoTokenizer.from_pretrained(model_id)
#model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

for i, layer in enumerate(base_model.model.layers):
    if getattr(layer, "linear_replacement", None):
        with open(f"/NBL/llm_variables/xlayer_{i}_weights.pkl", "rb") as f:
            w, b = pickle.load(f)
        layer.initialize_linear_from_pickle(w.to(torch.float16), b.to(torch.float16))


peft_model_id = "/NBL/lora-linear-replacement-slim"
config = PeftConfig.from_pretrained(peft_model_id)
#base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, peft_model_id)
model = model.merge_and_unload()  #  merges LoRA weights into the base weights

model.save_pretrained("/NBL/ds_nbl_12_lora")  # Save merged model for eval
