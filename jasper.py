import re
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from datasets import Dataset, load_dataset

MATH_DATASET_ID = "EleutherAI/hendrycks_math"
MODEL_ID = "mistralai/Mistral-7B-v0.1"
BATCH_SIZE = 2


def extract_boxed_text(text: str) -> str:
    """Extracts the last \\boxed{...} content from a string."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else None


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem_text: str,
    max_new_tokens: int = 256
) -> str:
    prompt = (
        "Answer the following math question and provide your final simplified numerical "
        "answer on its own line preceded by Answer:\n"
        f"Problem: {problem_text}\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
        )
    gen = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    )
    return gen.strip()


def construct_verification_prompt(gt: str, pred: str) -> str:
    return (
        "You are an expert math evaluator.\n\n"
        "Extract the final numerical answer from the model's prediction and compare it to the correct answer"
        "Allow for equivalent expressions (e.g., 0.5 and 1/2 are the same).\n"
        "Respond **concisely**, with your final verdict in a box:\n\n"
        "If the answer is correct, reply with \\boxed{Yes}\n\n"
        "If the answer is incorrect, reply with \\boxed{No}\n\n"
        "Correct Answer:\n"
        f"{gt}\n\n"
        "Model Prediction:\n"
        f"{pred}\n\n"
        "Is the model's answer mathematically correct? ONLY include the verdict.\n"
        "Verdict: "
    )


def batch_verify_with_llama4(
    verifier_model: AutoModelForCausalLM,
    verifier_processor: AutoProcessor,
    prompts: List[str],
    max_new_tokens: int = 256,
) -> List[str]:
    # 1) batch‑tokenize all prompts
    tok = verifier_processor.tokenizer
    enc = tok(
        prompts,
        return_tensors="pt",
        padding="longest",
        pad_to_multiple_of=8,
    ).to(verifier_model.device)

    # 2) generate on the full batch in one shot,
    #    using tokenizer.pad_token_id (an int)
    pad_id = verifier_processor.tokenizer.pad_token_id
    with torch.no_grad():
        outputs = verifier_model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            pad_token_id=pad_id,
            max_new_tokens=max_new_tokens,
        )

    # 3) strip off the shared prompt prefix
    prefix_len = enc.input_ids.shape[1]
    gen_slices = outputs[:, prefix_len:]

    # 4) decode and extract verdicts
    decodes = tok.batch_decode(gen_slices, skip_special_tokens=True)
    return [extract_boxed_text(d) for d in decodes]



def evaluate_with_llama_verifier(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    verifier_model: AutoModelForCausalLM,
    verifier_processor: AutoProcessor,
    dataset: Dataset,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, float]:
    correct = 0
    total = len(dataset)

    for start in tqdm(range(0, total, batch_size)):
        chunk = dataset.select(range(start, min(start + batch_size, total))).to_list()

        # Prepare all the generation + verification data
        prompts = []
        metadata = []  # (problem, gt_answer, pred_text)
        for ex in chunk:
            gt_box = extract_boxed_text(ex["solution"])
            if gt_box is None:
                continue
            pred = generate_answer(model, tokenizer, ex["problem"])
            prompts.append(construct_verification_prompt(gt_box, pred))
            metadata.append((ex["problem"], gt_box, pred))

        if not prompts:
            continue

        verdicts = batch_verify_with_llama4(
            verifier_model, verifier_processor, prompts
        )

        for (_, gt, pred), verdict in zip(metadata, verdicts):
            if verdict == "Yes":
                correct += 1
                print(f"✓ Correct | Pred: {pred} | GT: {gt}")
            else:
                print(f"✗ Incorrect | Pred: {pred} | GT: {gt}")

    accuracy = correct / total
    print(f"\nTotal Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    return {"accuracy": accuracy}


if __name__ == "__main__":
    ds = load_dataset(MATH_DATASET_ID, "prealgebra")["test"]

    # generation model
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to("cuda")
    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # LLaMA‑4 verifier
    llama_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    llama_processor = AutoProcessor.from_pretrained(
        "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    )

    evaluate_with_llama_verifier(
        base_model,
        base_tokenizer,
        llama_model,
        llama_processor,
        ds,
        batch_size=BATCH_SIZE,
    )
