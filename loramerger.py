# lora_merger_class.py

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# For detailed debugging:
# logging.getLogger().setLevel(logging.DEBUG)


class LoRAMerger:
    """
    Manages loading a base model, PEFT adapters, calculating combined deltas,
    and creating a merged model for inference.
    """

    def __init__(
        self,
        base_model_id: str,
        torch_dtype: torch.dtype = torch.float16,
        device: Optional[str] = None,
    ):
        self.base_model_id = base_model_id
        self.torch_dtype = torch_dtype
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.base_model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.peft_model: Optional[PeftModel] = None
        self.loaded_adapters: Dict[str, str] = {}
        self.combined_deltas: Optional[Dict[str, torch.Tensor]] = None
        self.merged_model: Optional[PreTrainedModel] = None
        logging.info(
            f"LoRAMerger initialized for base model: {self.base_model_id} on device: {self.device}"
        )

    # --- load_base method ---
    def load_base(
        self, device_map: Optional[Union[str, Dict]] = "auto"
    ) -> Tuple[
        Optional[PreTrainedModel], Optional[PreTrainedTokenizer]
    ]:  # Add return type hint
        """Loads the base model and tokenizer. Returns the loaded objects."""
        if self.base_model and self.tokenizer:
            logging.info("Base model and tokenizer already loaded.")
            # --- FIX 1: Return existing objects ---
            return self.base_model, self.tokenizer

        logging.info(f"Loading base model: {self.base_model_id}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            # trust_remote_code=True, # Uncomment if needed
        )
        logging.info(f"Loading tokenizer: {self.base_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            # trust_remote_code=True, # Uncomment if needed
        )
        if self.tokenizer.pad_token is None:
            logging.warning(
                "Tokenizer does not have a pad token. Setting to eos_token."
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logging.info("Base model and tokenizer loaded.")
        # --- FIX 2: Return newly loaded objects ---
        return self.base_model, self.tokenizer

    # --- load_adapters method --- (Keep the previous corrected version)
    def load_adapters(
        self, adapter_configs: List[Dict[str, str]], is_trainable: bool = False
    ):
        if self.base_model is None:
            # Try loading base if not already loaded? Or enforce calling order.
            # For now, enforce calling order.
            raise RuntimeError(
                "Base model must be loaded before loading adapters. Call `load_base()` first."
            )

        first_adapter_for_peft_model = self.peft_model is None

        for config in adapter_configs:
            peft_id = config.get("peft_id")
            adapter_name = config.get("adapter_name")
            if not peft_id or not adapter_name:
                raise ValueError(
                    "Each adapter config must contain 'peft_id' and 'adapter_name'."
                )
            if adapter_name in self.loaded_adapters:
                logging.warning(
                    f"Adapter '{adapter_name}' seems already loaded or name reused. Skipping loading again."
                )
                continue

            logging.info(
                f"Attempting to load adapter '{adapter_name}' from '{peft_id}'..."
            )
            try:
                if first_adapter_for_peft_model:
                    self.peft_model = PeftModel.from_pretrained(
                        self.base_model,
                        peft_id,
                        adapter_name=adapter_name,
                        is_trainable=is_trainable,
                    )
                    first_adapter_for_peft_model = False
                    logging.info(
                        f"Adapter '{adapter_name}' loaded. Model is now PeftModel."
                    )
                else:
                    self.peft_model.load_adapter(
                        peft_id,
                        adapter_name=adapter_name,
                        is_trainable=is_trainable,
                    )
                    logging.info(f"Adapter '{adapter_name}' loaded.")
                self.loaded_adapters[adapter_name] = peft_id
            except Exception as e:
                logging.error(
                    f"Failed to load adapter '{adapter_name}' from '{peft_id}': {e}",
                    exc_info=True,
                )
                raise

        logging.info(f"Current loaded adapters: {list(self.loaded_adapters.keys())}")
        if self.peft_model:
            for name, config in self.peft_model.peft_config.items():
                if isinstance(config, LoraConfig):
                    logging.info(
                        f"Adapter '{name}' config: rank={config.r}, alpha={config.lora_alpha}, targets={config.target_modules}"
                    )

    # --- calculate_combined_delta method --- (Keep the previous corrected version)
    def calculate_combined_delta(
        self,
        adapters_to_merge: List[str],
        weights: List[float],
        delta_compute_dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        if self.peft_model is None:
            raise RuntimeError(
                "Adapters must be loaded before calculating delta. Call `load_adapters()` first."
            )
        if len(adapters_to_merge) != len(weights):
            raise ValueError(
                "Length of adapters_to_merge and weights must be the same."
            )
        if not adapters_to_merge:
            raise ValueError("adapters_to_merge list cannot be empty.")

        peft_configs: Dict[str, LoraConfig] = {}
        all_target_modules = set()
        logging.info("Verifying adapters and configs for merging:")
        for name in adapters_to_merge:
            if name not in self.peft_model.peft_config:
                raise ValueError(
                    f"Adapter '{name}' requested for merge, but not found in the loaded PeftModel. Loaded: {list(self.loaded_adapters.keys())}"
                )
            config = self.peft_model.peft_config[name]
            if not isinstance(config, LoraConfig):
                raise TypeError(f"Adapter '{name}' is not a LoraConfig.")
            logging.info(
                f"  - Adapter '{name}': rank={config.r}, alpha={config.lora_alpha}, targets={config.target_modules}"
            )
            peft_configs[name] = config
            all_target_modules.update(config.target_modules)

        logging.info(
            f"Union of target modules from adapters to merge: {all_target_modules}"
        )
        logging.info(
            f"Calculating combined delta for adapters: {adapters_to_merge} with weights: {weights}"
        )

        calculated_deltas: Dict[str, torch.Tensor] = {}
        processed_layers_count = 0
        logging.debug("Starting iteration over base model modules...")
        with torch.no_grad():
            # Iterate over the *base_model* structure within the peft_model
            for name, module in self.peft_model.base_model.named_modules():
                is_target_type = isinstance(
                    module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv1d)
                )
                if is_target_type:
                    logging.debug(
                        f"Checking module: name='{name}', type={type(module)}"
                    )
                    has_lora_A = hasattr(module, "lora_A")
                    has_lora_B = hasattr(module, "lora_B")
                    logging.debug(
                        f"  - hasattr(lora_A): {has_lora_A}, hasattr(lora_B): {has_lora_B}"
                    )
                    layer_delta = None
                    found_delta_for_layer = False

                    for adapter_name, weight in zip(adapters_to_merge, weights):
                        config = peft_configs[adapter_name]
                        module_base_name = name.split(".")[-1]
                        if module_base_name in config.target_modules:
                            logging.debug(
                                f"    - Adapter '{adapter_name}' targets this module base name '{module_base_name}'."
                            )
                            lora_A_layer = (
                                module.lora_A.get(adapter_name, None)
                                if has_lora_A
                                else None
                            )
                            lora_B_layer = (
                                module.lora_B.get(adapter_name, None)
                                if has_lora_B
                                else None
                            )
                            if lora_A_layer is not None and lora_B_layer is not None:
                                logging.debug(
                                    f"      - Found lora_A and lora_B weights for '{adapter_name}' on module '{name}'."
                                )
                                lora_A = lora_A_layer.weight
                                lora_B = lora_B_layer.weight
                                if layer_delta is None:
                                    original_weight = module.weight
                                    layer_delta = torch.zeros_like(
                                        original_weight, dtype=delta_compute_dtype
                                    )
                                scaling = (
                                    config.lora_alpha / config.r
                                    if config.r != 0
                                    else 0.0
                                )
                                delta = (lora_B @ lora_A) * scaling * weight
                                layer_delta += delta.to(delta_compute_dtype)
                                found_delta_for_layer = True
                            else:
                                logging.warning(
                                    f"      - Adapter '{adapter_name}' targets '{module_base_name}', BUT lora_A/lora_B weights were NOT found attached to module '{name}'. Skipping delta for this adapter/layer."
                                )

                    if (
                        found_delta_for_layer
                        and layer_delta is not None
                        and torch.count_nonzero(layer_delta) > 0
                    ):
                        calculated_deltas[name] = layer_delta
                        processed_layers_count += 1
                        logging.debug(f"  - Stored combined delta for layer: '{name}'")

        logging.info(
            f"Finished calculating deltas. Found combined deltas for {len(calculated_deltas)} layers (processed {processed_layers_count} layers with non-zero delta)."
        )
        if not calculated_deltas:
            logging.warning(
                "Calculated deltas dictionary is empty! Check adapter target modules, PEFT loading, and logs."
            )
        self.combined_deltas = calculated_deltas
        return self.combined_deltas

    # --- merge_and_create_model method ---
    def merge_and_create_model(self, device_map: Optional[Union[str, Dict]] = "auto"):
        """
        Creates a new model by applying the calculated combined deltas.

        Args:
            device_map: Device map strategy for the final merged model.

        Returns:
            The merged PreTrainedModel instance. Also stores in `self.merged_model`.
        """
        if self.combined_deltas is None:
            raise RuntimeError(
                "Combined deltas have not been calculated. Call `calculate_combined_delta()` first."
            )

        # --- FIX 3: Simplify the empty delta case ---
        if not self.combined_deltas:
            logging.warning(
                "Calculated combined deltas are empty. Cannot merge. "
                "Returning a fresh base model instance instead."
            )
            # Directly load and return a new base model instance
            fresh_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=self.torch_dtype,
                device_map=device_map,
                # trust_remote_code=True, # Uncomment if needed
            )
            self.merged_model = fresh_model  # Store it if desired
            return self.merged_model
        # --- End of FIX 3 ---

        logging.info("Creating new model instance for merging...")
        # Load fresh base model structure
        self.merged_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            # trust_remote_code=True, # Uncomment if needed
        )
        logging.info("Applying combined deltas to the new model instance...")

        updated_layers = set()
        state_dict = self.merged_model.state_dict()

        with torch.no_grad():
            for name, delta_tensor in self.combined_deltas.items():
                weight_key = f"{name}.weight"
                if weight_key in state_dict:
                    original_weight = state_dict[weight_key]
                    delta = delta_tensor.to(
                        original_weight.device, dtype=original_weight.dtype
                    )
                    state_dict[weight_key] = original_weight + delta
                    updated_layers.add(name)
                    logging.debug(
                        f"  - Prepared updated weight for layer state_dict key: {weight_key}"
                    )
                else:
                    logging.warning(
                        f"  - Delta calculated for layer '{name}', but corresponding weight key '{weight_key}' not found in target model state_dict! Delta will NOT be applied for this layer."
                    )

        self.merged_model.load_state_dict(state_dict)

        if not updated_layers:
            logging.error(
                "Merge process completed, but NO layers were updated in the state_dict. Check for state_dict key mismatches."
            )
        else:
            logging.info(
                f"Applied combined deltas by updating state_dict for {len(updated_layers)} layers."
            )
        logging.info("Merged model is ready (no longer LoRA).")
        return self.merged_model

    # --- get_model_for_inference method --- (Keep the previous corrected version)
    def get_model_for_inference(
        self, use_merged: bool = True, active_adapter: Optional[str] = None
    ) -> PreTrainedModel:
        if use_merged:
            if self.merged_model is None:
                raise RuntimeError(
                    "Merged model requested, but not created yet. Call `merge_and_create_model()`."
                )
            return self.merged_model
        else:
            if self.peft_model is None:
                raise RuntimeError("PeftModel requested, but no adapters loaded yet.")
            if active_adapter:
                if active_adapter not in self.loaded_adapters:
                    raise ValueError(
                        f"Adapter '{active_adapter}' requested but not loaded."
                    )
                if active_adapter not in self.peft_model.peft_config:
                    raise ValueError(
                        f"Adapter '{active_adapter}' not found in peft_model's config. Available: {list(self.peft_model.peft_config.keys())}"
                    )
                logging.info(
                    f"Setting active adapter on PeftModel to: {active_adapter}"
                )
                self.peft_model.set_adapter(active_adapter)
            elif not self.peft_model.active_adapter:
                first_adapter = next(iter(self.loaded_adapters.keys()), None)
                if first_adapter:
                    logging.warning(
                        f"No active adapter set on PeftModel. Activating the first loaded one: '{first_adapter}'"
                    )
                    self.peft_model.set_adapter(first_adapter)
                else:
                    raise RuntimeError(
                        "PeftModel requested, but no adapters are loaded or active."
                    )
            return self.peft_model

    # --- run_inference method --- (Keep the previous corrected version)
    def run_inference(
        self,
        prompts: List[str],
        use_merged_model: bool = True,
        active_adapter_for_peft: Optional[str] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.2,
        **generate_kwargs: Any,
    ) -> None:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call `load_base()` first.")
        model_to_use = self.get_model_for_inference(
            use_merged=use_merged_model, active_adapter=active_adapter_for_peft
        )
        active_adapter_name = (
            model_to_use.active_adapter
            if hasattr(model_to_use, "active_adapter")
            else "N/A (Merged)"
        )
        logging.info(
            f"\n--- Running Inference using {'Merged Model' if use_merged_model else f'PEFT Model (adapter: {active_adapter_name})'} ---"
        )

        if hasattr(model_to_use, "hf_device_map") and model_to_use.hf_device_map:
            try:
                model_device = next(iter(model_to_use.hf_device_map.values()))
                logging.info(
                    f"Using device_map: {model_to_use.hf_device_map}. Placing inputs on {model_device}."
                )
            except Exception:
                logging.warning(
                    "Could not determine primary device from device_map. Using default device: {self.device}"
                )
                model_device = self.device
        elif hasattr(model_to_use, "device"):
            model_device = model_to_use.device
            logging.info(f"Using model device: {model_device}")
        else:
            try:
                model_device = next(model_to_use.parameters()).device
                logging.info(f"Using device from model parameters: {model_device}")
            except StopIteration:
                logging.error("Could not determine model device. Cannot run inference.")
                return

        for prompt in prompts:
            logging.info(f"\nPrompt: {prompt}")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(
                model_device
            )
            with torch.no_grad():
                outputs = model_to_use.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True,
                    **generate_kwargs,
                )
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info(f"Generated Output:\n{decoded_output}")

    # --- cleanup method --- (Keep the previous version)
    def cleanup(self):
        """Releases model references to free memory."""
        logging.info("Cleaning up model references...")
        del self.base_model
        del self.peft_model
        del self.merged_model
        self.base_model = None
        self.peft_model = None
        self.merged_model = None
        self.combined_deltas = None
        self.loaded_adapters = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleanup complete.")


# --- Example Usage --- (Should now work without the TypeError)
if __name__ == "__main__":
    BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"
    ADAPTER_CONFIGS = [
        {"peft_id": "predibase/gsm8k", "adapter_name": "gsm8k"},
        {"peft_id": "predibase/magicoder", "adapter_name": "magicoder"},
    ]
    ADAPTERS_TO_MERGE = ["gsm8k", "magicoder"]
    MERGE_WEIGHTS = [0.5, 0.5]

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_DTYPE = (
        torch.bfloat16
        if DEVICE == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    # Set logging to DEBUG to trace delta calculation
    logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize the merger
        merger = LoRAMerger(
            base_model_id=BASE_MODEL_ID,
            torch_dtype=MODEL_DTYPE,
            device=DEVICE,
        )

        merger.load_base(device_map="auto")
        merger.load_adapters(adapter_configs=ADAPTER_CONFIGS)

        combined_deltas_result = merger.calculate_combined_delta(
            adapters_to_merge=ADAPTERS_TO_MERGE,
            weights=MERGE_WEIGHTS,
            delta_compute_dtype=torch.float32,
        )
        logging.info(
            f"Number of layers in combined_deltas: {len(combined_deltas_result)}"
        )

        merged_model = merger.merge_and_create_model(device_map="auto")

        if merged_model and merger.tokenizer:
            test_prompts = [
                "Janet has 5 apples. She buys 3 more bags of apples, with 4 apples in each bag. How many apples does she have now?",
                "Write a python function to calculate the factorial of a number recursively.",
            ]
            merger.run_inference(
                prompts=test_prompts,
                use_merged_model=True,
                max_new_tokens=256,
            )
        else:
            logging.error(
                "Merged model creation likely failed (check logs). Skipping inference."
            )

    except Exception as e:
        logging.error(f"An error occurred in the merging process: {e}", exc_info=True)
    finally:
        logging.getLogger().setLevel(logging.INFO)  # Reset log level
