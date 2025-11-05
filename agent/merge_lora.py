import argparse
from peft import PeftModel, PeftConfig
from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
import os

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into a base model and save as a full checkpoint")
    
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path or name of the base model (e.g., 'LLM4Binary/llm4decompile-1.3b-v1.5' or '/path/to/base/model')"
    )
    
    parser.add_argument(
        "--lora_adapter", 
        type=str, 
        required=True,
        help="Path to the LoRA adapter directory (e.g., './lora_adapter')"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the merged model will be saved (e.g., './merged_model')"
    )
    
    parser.add_argument(
        "--save_tokenizer",
        action="store_true",
        help="Whether to also save the tokenizer from the base model"
    )
        
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.lora_adapter):
        raise ValueError(f"LoRA adapter path does not exist: {args.lora_adapter}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the base model
    print(f"Loading base model from: {args.base_model}")
    base_model = AutoModelForImageTextToText.from_pretrained(args.base_model)
    
    # Load the LoRA adapter to create a PeftModel
    print(f"Loading LoRA adapter from: {args.lora_adapter}")
    model_to_merge = PeftModel.from_pretrained(base_model, args.lora_adapter)
    
    # Merge LoRA weights and unload the adapter structure
    print("Merging LoRA weights...")
    merged_model = model_to_merge.merge_and_unload()
    
    # Save the fully merged model
    print(f"Saving merged model to: {args.output_dir}")
    merged_model.generation_config.do_sample = True
    merged_model.save_pretrained(
        args.output_dir,
        safe_serialization=True
    )
    
    # Save the tokenizer if requested
    if args.save_tokenizer:
        print("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.save_pretrained(args.output_dir)
    
    print("Merge complete!")
    print(f"Full checkpoint saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
