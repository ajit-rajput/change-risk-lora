## Note: The above code is a complete script for merging LoRA weights into a base language model. 
# It loads the base model and tokenizer, applies the LoRA adapter, merges the weights, and saves the resulting model.
## This is useful for deploying a single model file without needing to load separate LoRA adapters.
## Once the merge is successful, follow the steps
# -- Convert merged model to GGUF by cloning and using the gguf-convert script from https://github.com/ggerganov/llama.cpp
## -- in current directory ==> git clone https://github.com/ggerganov/llama.cpp.git
## -- then run the conversion script as:
## -- python llama.cpp/convert-hf-to-gguf.py training/merged_model --outfile change-risk-lora.gguf
## once the script runs successfully, you will get change-risk-lora.gguf file in current directory.


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "training/lora_adapter"
MERGED_PATH = "training/merged_model"

def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Load base model WITHOUT accelerate dispatch
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False
    )

    # Force model onto CPU explicitly
    model.to("cpu")

    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        model,
        LORA_PATH,
        is_trainable=False
    )

    # Merge LoRA weights into base model
    model = model.merge_and_unload()

    # Save merged model
    model.save_pretrained(MERGED_PATH)
    tokenizer.save_pretrained(MERGED_PATH)

    print("LoRA successfully merged into base model")

if __name__ == "__main__":
    main()

