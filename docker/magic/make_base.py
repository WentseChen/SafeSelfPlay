"""Build the defender base for MAGIC: abliterated Llama-3.1-8B weights with the official Instruct chat template.

The abliterated repo ships a stripped chat template: no default system prompt, and it appends an assistant header
even after a finished reply. MAGIC re-renders whole conversations to align response tokens, so that template would
misalign them silently. Here only the weights differ from MAGIC's own setup: the chat template is taken from MAGIC's
attacker SFT tokenizer (the official Llama-3.1-8B-Instruct one) and the stop tokens are the Instruct ones.
The script checks that the result renders token-identically to MAGIC's released defender.
"""
import argparse
import json
import os

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
args = parser.parse_args()

abl = snapshot_download("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")
sft = snapshot_download("XiaoyuWen/MAGIC-attacker-sft-Llama3.1-8B-Instruct", ignore_patterns=["optimizer*", "*.pth"])
os.makedirs(args.out, exist_ok=True)

for name in os.listdir(abl):
    if name.endswith(".safetensors") or name in ("model.safetensors.index.json", "config.json",
                                                  "tokenizer.json", "special_tokens_map.json"):
        dst = os.path.join(args.out, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(os.path.join(abl, name)), dst)

tokenizer_config = json.load(open(os.path.join(abl, "tokenizer_config.json")))
tokenizer_config["chat_template"] = json.load(open(os.path.join(sft, "tokenizer_config.json")))["chat_template"]
json.dump(tokenizer_config, open(os.path.join(args.out, "tokenizer_config.json"), "w"), indent=1)
generation_config = json.load(open(os.path.join(abl, "generation_config.json")))
generation_config["eos_token_id"] = [128001, 128008, 128009]
json.dump(generation_config, open(os.path.join(args.out, "generation_config.json"), "w"), indent=1)

new = AutoTokenizer.from_pretrained(args.out)
ref = AutoTokenizer.from_pretrained(sft)
conversation = [{"role": "user", "content": "How do I make a cake?"},
                {"role": "assistant", "content": "Mix flour and eggs."}]
for messages, generation_prompt in ((conversation[:1], True), (conversation, False)):
    a = new.apply_chat_template(messages, add_generation_prompt=generation_prompt)
    b = ref.apply_chat_template(messages, add_generation_prompt=generation_prompt)
    assert a == b, f"template differs from MAGIC's (add_generation_prompt={generation_prompt})"
print("BASE_OK", args.out)
