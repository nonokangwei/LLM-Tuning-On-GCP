# Example usage:
# python3 merge_peft.py --base_model=models/base_model --peft_model=models/lora_model --saved_path=models/merged_model


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--peft_model", type=str)
    parser.add_argument("--saved_path", type=str)

    return parser.parse_args()

def main():
    args = get_args()

    print(f"[1/4] Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"[2/4] Loading adapter: {args.peft_model}")
    model = PeftModel.from_pretrained(base_model, args.peft_model, device_map="auto")
    
    print("[3/4] Merge base model and adapter")
    model = model.merge_and_unload()
    model.save_pretrained(args.saved_path)
    tokenizer.save_pretrained(args.saved_path)
    
    print("[4/4] Save merged model to local path")

if __name__ == "__main__" :
    main()