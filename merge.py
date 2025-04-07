import copy  # To create a separate model instance for inference

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mistral-7B-v0.1"
peft_model_gsm8k_id = "predibase/gsm8k"
peft_model_magicoder_id = "predibase/magicoder"
adapter_name_gsm8k = "gsm8k"
adapter_name_magicoder = "magicoder"

device = "cuda" if torch.cuda.is_available() else "cpu"
delta_compute_dtype = torch.float32
final_model_dtype = torch.bfloat16 if device == "cuda" else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=final_model_dtype,  # Load with target dtype
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model_with_adapters = PeftModel.from_pretrained(
    base_model,
    peft_model_gsm8k_id,
    adapter_name=adapter_name_gsm8k,
    is_trainable=False,
)
model_with_adapters.load_adapter(
    peft_model_magicoder_id, adapter_name=adapter_name_magicoder, is_trainable=False
)

config1 = model_with_adapters.peft_config[adapter_name_gsm8k]
config2 = model_with_adapters.peft_config[adapter_name_magicoder]

if config1.target_modules != config2.target_modules:
    # If they target different modules, logic needs adjustment
    # For simplicity, assume they target the same set or handle appropriately
    print("Warning: Adapters target different modules. Combining common targets.")
    target_modules = set(config1.target_modules) & set(config2.target_modules)
    if not target_modules:
        raise ValueError("Adapters have no common target modules to combine.")
else:
    target_modules = set(config1.target_modules)  # Should be a set

print(f"Target modules for combining: {target_modules}")

scaling1 = config1.lora_alpha / config1.r if config1.r != 0 else 0.0
scaling2 = config2.lora_alpha / config2.r if config2.r != 0 else 0.0
weight1 = 0.5  # Weight for adapter 1 contribution
weight2 = 0.5  # Weight for adapter 2 contribution

combined_deltas = {}  # Dictionary to store {layer_name: combined_delta_tensor}

with torch.no_grad():
    # Iterate over the modules of the model *with adapters* to find LoRA weights
    # Use the underlying base_model structure for module names if PEFT wraps them
    # Need to adjust path prefixes like 'base_model.model.' based on inspection
    module_prefix = "base_model.model."  # Common prefix, adjust if needed!

    for name, module in model_with_adapters.named_modules():
        # Strip prefix if needed to match target_modules notation
        clean_name = name.replace(module_prefix, "")
        module_base_name = clean_name.split(".")[-1]  # e.g. 'q_proj', 'v_proj'

        # Check if this module type in this layer was targeted
        if module_base_name in target_modules and isinstance(module, torch.nn.Linear):
            print(f"Processing layer: {name} ({clean_name})")

            # Initialize combined delta for this layer
            # Use the shape of the original weight from the base model
            original_weight = base_model.get_submodule(name).weight
            current_delta = torch.zeros_like(original_weight, dtype=delta_compute_dtype)

            # --- Calculate delta for adapter 1 ---
            lora_A1_layer = module.lora_A.get(adapter_name_gsm8k, None)
            lora_B1_layer = module.lora_B.get(adapter_name_gsm8k, None)
            if lora_A1_layer is not None and lora_B1_layer is not None:
                lora_A1 = lora_A1_layer.weight
                lora_B1 = lora_B1_layer.weight
                # Calculate B@A
                delta1 = (lora_B1 @ lora_A1) * scaling1 * weight1
                current_delta += delta1.to(delta_compute_dtype)  # Add to combined delta
                print(f"  - Added delta from {adapter_name_gsm8k}")

            # --- Calculate delta for adapter 2 ---
            lora_A2_layer = module.lora_A.get(adapter_name_magicoder, None)
            lora_B2_layer = module.lora_B.get(adapter_name_magicoder, None)
            if lora_A2_layer is not None and lora_B2_layer is not None:
                lora_A2 = lora_A2_layer.weight
                lora_B2 = lora_B2_layer.weight
                # Calculate B@A
                delta2 = (lora_B2 @ lora_A2) * scaling2 * weight2
                current_delta += delta2.to(delta_compute_dtype)  # Add to combined delta
                print(f"  - Added delta from {adapter_name_magicoder}")

            # Store the final combined delta for this layer name
            if torch.count_nonzero(current_delta) > 0:
                combined_deltas[name] = current_delta  # Store with the full name key
                print(f"  - Stored combined delta for {name}")
            else:
                print(f"  - Skipping storage for {name} (zero delta)")

print(
    f"\nFinished calculating deltas. Found combined deltas for {len(combined_deltas)} layers."
)

del model_with_adapters
del base_model
torch.cuda.empty_cache()

# Create a fresh copy of the base model structure, potentially loading directly to target device
inference_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=final_model_dtype,
    device_map="auto",
)

print("Applying combined deltas to the inference model...")
with torch.no_grad():
    # Iterate through the layers of the *new* inference model
    for name, module in inference_model.named_modules():
        if name in combined_deltas:  # Check if we calculated a delta for this layer
            # Get the original weight (W0) from this instance
            W0 = module.weight.clone()
            # Get the calculated combined delta
            delta = combined_deltas[name].to(
                W0.device, dtype=W0.dtype
            )  # Match device/dtype

            # Apply the delta: W_final = W0 + delta
            module.weight.copy_(W0 + delta)
            print(f"  - Applied delta to layer: {name}")

print("\n--- Testing the combined model ---")

# Ensure tokenizer pads correctly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt_gsm8k = "Janet has 5 apples. She buys 3 more bags of apples, with 4 apples in each bag. How many apples does she have now?"
prompt_magicoder = (
    "Write a python function to calculate the factorial of a number recursively."
)

for prompt in [prompt_gsm8k, prompt_magicoder]:
    print(f"\nPrompt: {prompt}")
    # Use the inference_model's device
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(
        inference_model.device
    )

    with torch.no_grad():
        outputs = inference_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.2,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated Output:\n{decoded_output}")

save_dir = "./weights/fully_merged_model"
print(f"\nSaving fully merged model to {save_dir}")
inference_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
