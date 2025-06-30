from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import HfPairwiseJudge, OpenAIPairwiseJudge
import torch

# Parameters
model_path = '/home/nstrang2/scratch/FinishedLLMs/Meta-Llama-3-8B-Instruct-OnlineDPO-WIM-Zeta0.0'

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
model.eval()

# Generate the model completions
num_examples = 1000

# Load the dataset
dataset = load_dataset("/home/nstrang2/scratch/Datasets/tldr/", split="validation")
if num_examples is not None:
    dataset = dataset.select(range(num_examples))

# Extract the prompts and reference completions
raw_prompts = dataset["prompt"]

# Run the model completions
model_completions = []
for i, raw_prompt in enumerate(raw_prompts):
    # Extract the correct content
    prompt = raw_prompt

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            top_p=0.95,
            temperature=0.25,
            max_new_tokens=1000
        )
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    model_completions.append(completion)
    print(f'Completion: {i+1} / {num_examples}')

# Save the model completions
with open("completions.txt", "w", encoding="utf-8") as f:
    for completion in model_completions:
        f.write("{\n" + completion.strip() + "\n}\n\n")