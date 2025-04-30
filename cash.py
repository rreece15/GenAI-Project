import re
from typing import Dict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from datasets import Dataset, load_dataset

MATH_DATASET_ID = "EleutherAI/hendrycks_math"
MODEL_ID        = "mistralai/Mistral-7B-v0.1"


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


def verify_with_llama4_llm(
    verifier_model: AutoModelForCausalLM,
    verifier_processor: AutoProcessor,
    ground_truth: str,
    prediction_text: str,
    max_new_tokens: int = 128
) -> bool:
    """Runs a single LLaMA‑4 check, returns True if verdict is Yes."""
    prompt = (
        "You are an expert math evaluator.\n\n"
        "Extract the final numerical answer from the model's prediction and compare it to the correct answer"
        "Allow for equivalent expressions (e.g., 0.5 and 1/2 are the same).\n"
        "Respond **concisely**, with your final verdict in a box:\n\n"
        "If the answer is correct, reply with \\boxed{Yes}\n\n"
        "If the answer is incorrect, reply with \\boxed{No}\n\n"
        "Correct Answer:\n"
        f"{ground_truth}\n\n"
        "Model Prediction:\n"
        f"{prediction_text}\n\n"
        "Is the model's answer mathematically correct? ONLY include the verdict.\n"
        "Verdict: "
    )

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    inputs = verifier_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(verifier_model.device)

    with torch.no_grad():
        outputs = verifier_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=verifier_processor.tokenizer.pad_token_id
        )

    resp = verifier_processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )[0]
    verdict = extract_boxed_text(resp)
    return verdict == "Yes"


def evaluate_with_llama_verifier(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    verifier_model: AutoModelForCausalLM,
    verifier_processor: AutoProcessor,
    dataset: Dataset
) -> Dict[str, float]:
    correct = 0
    total   = len(dataset)

    for example in tqdm(dataset, total=total):
        problem  = example["problem"]
        solution = example["solution"]
        gt_box   = extract_boxed_text(solution)
        if gt_box is None:
            continue

        pred_text = generate_answer(model, tokenizer, problem)
        is_correct = verify_with_llama4_llm(
            verifier_model, verifier_processor, gt_box, pred_text
        )

        if is_correct:
            correct += 1
            print(f"✓ Correct | Pred: {pred_text} | GT: {gt_box}")
        else:
            print(f"✗ Incorrect | Pred: {pred_text} | GT: {gt_box}")

    accuracy = correct / total
    print(f"\nTotal Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    return {"accuracy": accuracy}


if __name__ == "__main__":
    ds = load_dataset(MATH_DATASET_ID, "prealgebra")["test"]

    # generation model
    base_model     = AutoModelForCausalLM.from_pretrained(MODEL_ID).to("cuda")
    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # LLaMA‑4 verifier
    llama_model     = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    llama_processor = AutoProcessor.from_pretrained(
        "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    )

    evaluate_with_llama_verifier(
        base_model,
        base_tokenizer,
        llama_model,
        llama_processor,
        ds
    )
