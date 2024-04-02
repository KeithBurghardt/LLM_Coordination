import pandas as pd
from utils import *
import json
from nltk.tokenize import word_tokenize
import argparse

decoding_args = OpenAIDecodingArguments(
    max_tokens=512
)

# truncate too long message
def truncate_sentence(sentence, max_tokens=256):
    tokens = word_tokenize(sentence)
    if len(tokens) <= max_tokens:
        return sentence
    truncated_tokens = tokens[:max_tokens]
    truncated_sentence = ' '.join(truncated_tokens)
    truncated_sentence = truncated_sentence + " ..."
    return truncated_sentence

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="prompt.txt")
parser.add_argument("--input", default="sample.json")
parser.add_argument("--output", default="result.json")
args = parser.parse_args()

prompt_file_path = args.prompt
with open(prompt_file_path, 'r') as file:
    prompt = file.read()

df = pd.read_json(args.input)
text_list = df["text"].tolist()

# preprocess prompt list
prompt_lst = []
for text in text_list:
    trunc_text = truncate_sentence(text, 512)
    prompt_lst.append(prompt + "\nTweet: " +trunc_text)

# prompt llm
results, finish_reason_lst, token_count, cost = openai_complete(prompt_lst, decoding_args, "gpt-4")
print("cost: {}".format(cost))

result_df = pd.DataFrame({'text': text_list, 'label': results})

# Save result
records = result_df.to_dict(orient='records')
with open(args.output, 'w') as json_file:
    json_file.write(json.dumps(records, indent=4))


