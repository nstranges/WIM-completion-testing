import re
from trl import HfPairwiseJudge, OpenAIPairwiseJudge
from datasets import load_dataset
import json
import os
import time

# The model testing the win rate. Using reference or another model
judge_model = 'gpt-4o-mini'
using_ref_completions = False

# Load the answers
with open("../Completions/TLDR/wim0.0-completions.txt", "r", encoding="utf-8") as f:
    raw = f.read()
model_completions = re.findall(r"\{\n(.*?)\n\}", raw, flags=re.DOTALL)
print('Loaded the completions')

# Load the dataset
num_examples = 1000
dataset = load_dataset("trl-lib/tldr", split="validation")
if num_examples is not None:
    dataset = dataset.select(range(num_examples))

# Reference completions
prompts = dataset["prompt"]

if using_ref_completions:
    reference_completions = dataset["completion"]
else:
    with open("../Completions/TLDR/base-completions.txt", "r", encoding="utf-8") as f:
        raw = f.read()
    reference_completions = re.findall(r"\{\n(.*?)\n\}", raw, flags=re.DOTALL)
    print('Loaded the other model completions')

print('Loaded the datasets')

# Load the openai API token
with open('openai_token.json', 'r') as config_file:
    config = json.load(config_file)

os.environ["OPENAI_API_KEY"] = config['token']

# Judge the outputs
if "gpt" in judge_model:
    judge = OpenAIPairwiseJudge(judge_model)
else:
    judge = HfPairwiseJudge(judge_model)

# Get all of the completions
batch_size = 5
delayed_results = []
total_done = 0
completions = [[c0, c1] for c0, c1 in zip(reference_completions, model_completions)]

# Loop through these in batches
for i in range(0, len(prompts), batch_size):
    prompt_batch = prompts[i:i + batch_size]
    completion_batch = completions[i:i + batch_size]
    
    result = judge.judge(prompt_batch, completion_batch)
    delayed_results.extend(result)

    total_done += batch_size
    print(f'Done {total_done} / {num_examples}')
    time.sleep(4)

best_idxs = delayed_results
model_win_rate = best_idxs.count(1) / len(best_idxs)
print(f"Model win rate: {model_win_rate * 100:.2f}%")