import os
import random
import numpy as np
import argparse
import pandas as pd

import torch
import datasets
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    T5Tokenizer,
    T5ForConditionalGeneration
)
from peft import get_peft_model, AdaLoraConfig, TaskType
from finetune import get_supported_backbones
from peft import PeftModel

def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t").replace('%20','  ')

def single_infer(model, tokenizer, inputs, max_length, max_new_tokens):
    # prompt = postprocess(inputs)
    prompt = f"{inputs}\n答："
    inp = tokenizer(text=[prompt], max_length=max_length, truncation=True, padding=True, return_tensors="pt")
    inp = inp.to("cuda")
    outputs = model.generate(**inp, max_new_tokens=max_new_tokens)
    out_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return {"inputs": prompt, "raw": out_text[0].strip().replace('\n','\\n'), "result": out_text[0][len(prompt):].strip().replace('\n','')}

def load_file(file_path):
    ffmt = file_path.strip().split('.')[-1]
    if ffmt in ("xls", "xlsx"):
        df = pd.read_excel(file_path)
    elif ffmt in ("csv"):
        df = pd.read_csv(file_path)
    return df

def inference(args):
    # get tokenizer and model type
    tokenizer_cls, model_cls = get_supported_backbones(args.model_type)
    # tokenizer
    print(f"loading tokenizer")
    tokenizer = tokenizer_cls.from_pretrained(args.model_dir, trust_remote_code=True)
    # model
    print(f"loading tuned model from {args.model_dir}")
    model = model_cls.from_pretrained(args.model_dir, trust_remote_code=True, device_map={"":0})
    if args.lora_ckpt:
        print(f"loading lora checkpoints {args.lora_ckpt}")
        model = PeftModel.from_pretrained(model, args.lora_ckpt)
        model = model.merge_and_unload()
    model.eval()
    # start simple infer
    test_df = load_file(args.file_path)
    print(f"test file loading, data size {test_df.size}")
    with torch.no_grad():
        results = [single_infer(model, tokenizer, inputs[args.input_key], args.max_length, args.max_new_tokens) for idx, inputs in test_df.iterrows()]
    # write down results
    print(f"done, writing {len(results)} results")
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_dir)
    print("data frame writing done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("simple chat bot inference")
    parser.add_argument("--model_dir", type=str, default="./")
    parser.add_argument("--model_type", type=str, default="chatglm2")
    parser.add_argument("--lora_ckpt", type=str, default="")
    parser.add_argument("--input_key", type=str, default="inputs")
    parser.add_argument("--file_path", type=str, default="test.csv")
    parser.add_argument("--output_dir", type=str, default="result.csv")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    args=parser.parse_args()
    print(args)
    inference(args)