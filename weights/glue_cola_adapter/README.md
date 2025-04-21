---
library_name: peft
base_model: mistralai/Mistral-7B-v0.1
pipeline_tag: text-generation
---
Description: Grammar and syntax acceptability\
Original dataset: https://huggingface.co/datasets/glue/viewer/cola \
---\
Try querying this adapter for free in Lora Land at https://predibase.com/lora-land! \
The adapter_category is Academic Benchmarks and the name is Linguistic Acceptability (CoLA)\
---\
Sample input: Determine if the sentence below is syntactically and semantically correct. If it is syntactically and semantically correct, respond "1". Otherwise, respond "0".\n\nSentence: Every senator seems to become more corrupt, as he talks to more lobbyists.\n\nLabel: \
---\
Sample output: 1\
---\
Try using this adapter yourself!
```
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mistral-7B-v0.1"
peft_model_id = "predibase/glue_cola"

model = AutoModelForCausalLM.from_pretrained(model_id)
model.load_adapter(peft_model_id)
```