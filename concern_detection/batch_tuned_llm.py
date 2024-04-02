from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd
from tqdm import tqdm
import warnings
import json
import nltk
from nltk.tokenize import word_tokenize
import argparse
nltk.download('punkt')
warnings.filterwarnings("ignore")

def truncate_sentence(sentence, max_tokens=256):
    tokens = word_tokenize(sentence)
    if len(tokens) <= max_tokens:
        return sentence
    truncated_tokens = tokens[:max_tokens]
    truncated_sentence = ' '.join(truncated_tokens)
    truncated_sentence = truncated_sentence + " ..."
    return truncated_sentence

# prompt format - llama2
def build_prompt(system_prompt, user_message):
    if system_prompt is not None:
        SYS = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>"
    else:
        SYS = ""
    CONVO = ""
    SYS = "<s>" + SYS
    CONVO += f"[INST] {user_message} [/INST]"
    return SYS + CONVO

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="prompt.txt")
parser.add_argument("--input", default="data.json")
parser.add_argument("--model", default="tuned_model/")
parser.add_argument("--output", default="labeled_data.json")
args = parser.parse_args()

print("building prompts ...")
df = pd.read_json(args.input)
test_list = df["text"].tolist()
input_list = [truncate_sentence(text, 256) for text in test_list]
sys_prompt = args.prompt
prompt_list = []
for user_message in input_list:
    prompt = build_prompt(sys_prompt, user_message)
    prompt_list.append(prompt)

print("loading model and tokenizer ...")
model = args.model
tokenizer = AutoTokenizer.from_pretrained(model, max_length=512)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map=0
)

print("inference ...")
predict_results = []
def process_group(group):
    sequences = pipeline(
        group,
        do_sample=True,
        top_k=50,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=100,
        return_full_text=False
    )
    for seq in sequences:
        predict_results.append(seq[0]["generated_text"])
step = 4
for i in tqdm(range(0, len(prompt_list), step)):
    group = prompt_list[i:i+step]
    process_group(group)
df["predict_raw"] = predict_results

records = df.to_dict(orient='records')
with open(args.output, 'w') as json_file:
    json_file.write(json.dumps(records, indent=4))



