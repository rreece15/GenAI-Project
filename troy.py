import re

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

def extract_boxed_answer(text):
    """
    Extracts the content of the last encountered \boxed{...} instance from a given string.
    Returns None if no such instance is found.
    """
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else None
ground_truth = "4"
predicted_answer="""
## Expert Answer

Answer: 10

## Explanation

The first spin is 20.

The factors of 20 are 1, 2, 4, 5, 10, and 20.

The second spin is 1.

The factors of 1 are 1.

The third spin is 2.

The factors of 2 are 1 and 2.

The fourth spin is 4.

The factors of 4 are 1, 2, and 
"""

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"""You are an expert math evaluator.

1. Extract the final numerical answer from the model's prediction.
2. Compare it to the correct answer, allowing for equivalent expressions (e.g., 0.5 and 1/2 are the same).
3. Respond **concisely**, with your final verdict in a box:

If the answer is correct, reply with \\boxed{{Yes}}

If the answer is incorrect, reply with \\boxed{{No}}

Correct Answer:  
{ground_truth}

Model Prediction:  
{predicted_answer}

Is the model's answer mathematically correct? ONLY include a brief one-line explanation and the verdict."""
            },
        ]
    },
]



inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)


outputs = model.generate(**inputs, max_new_tokens=256)

response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
print(response)

verdict = extract_boxed_answer(response)

if not (verdict is None):
    print(f"Verdict: {verdict}")

