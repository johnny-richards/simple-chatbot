import os
import random
import numpy as np
import argparse
import pandas as pd

import torch
import datasets
from transformers import (
    AutoModel,
    AutoModelForCasualLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    T5Tokenizer,
    T5ForConditionalGeneration
)
from peft import get_peft_model, AdaLoraConfig, TaskType
import bitsandtypes as bnb

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def tokenize(text, tokenizer, max_length, add_eos_token=True):
    result = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    if (result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) <  max_length
        and add_eos_token):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result

def preprocess(example, **kwargs):
    max_length = kwargs.get("max_length", 128)
    tokenizer = kwargs[tokenizer]
    train_on_inputs = kwargs.get("train_on_inputs", False)
    add_eos_token = kwargs.get("add_eos_tokens", True)
    # get prompt
    prompt = example["instruction"]
    response = example["output"]
    text = f"{prompt}{response}"
    tokenized_inp = tokenize(text, tokenizer, max_length, add_eos_token)
    if not train_on_inputs:
        tokenized_prompt = tokenize(prompt, tokenizer, max_length, add_eos_token)
        prompt_tokens_len = len(tokenized_prompt)
        tokenized_inp["labels"] = [-100] * prompt_tokens_len + tokenized_inp["labels"][prompt_tokens_len:]
    return tokenized_inp

def find_all_linear_names(model, bits=16):
    cls = bnb.nn.Linear4bit if bits==4 else (bnb.nn.Linear8itLt if bits==8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    # for mixture precisions
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

def get_supported_backbones(backbone_type):
    supported_dict = {
        't5': (T5Tokenizer, T5ForConditionalGeneration),
        'chatglm2': (AutoTokenizer, AutoModel)
    }
    return supported_dict.get(backbone_type, (AutoTokenizer, AutoModel))

def get_finetuning_model(model_cls, model_type, model_dir, **kwargs):
    device_map = {"": int(os.environ.get("LOCAL_RANK")) or 0}
    model = model_cls.from_pretrained(model_dir, trust_remote_code=True,
                                      device_map=device_map)
    lora_strategy = kwargs.get("lora", "")
    if lora_strategy == "":
        return model
    # only applicable if you choose to use lora
    if lora_strategy == "adalora":
        peft_config = AdaLoraConfig(
            task_type = TaskType.CasualLM,
            inference_model=False,
            r = kwargs["lora_r"],
            lora_alpha = kwargs["lora_alpha"],
            lora_dropout = kwargs["lora_dropout"]
            target_modules = find_all_linear_names(model, kwargs["bit"]) if model_type in ["chatglm2"] else ["q", "v"]
        )
    else:
        raise NotImplementedError
    peft_model = get_peft_model(model, peft_config)
    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    print("Lora is using, trainable paramters are:")
    peft_model.print_trainable_paramters()
    return peft_model

def get_dataset(data_path, tokenizer, max_length):
    df_data = pd.read_csv(data_path)
    # construct
    cur_dataset = datasets.Dataset.from_pandas(df_data)
    # mapping
    cur_data = cur_dataset.map(preprocess, fn_kwargs={"tokenizer":tokenizer,
                               "max_length": max_length})
    # return
    return cur_data

def main(args):
    # get model type
    print("tokenizer preparation")
    tokenizer_cls, model_cls = get_supported_backbones(args.model_type)
    tokenizer = tokenizer_cls.from_pretrained(args.model_dir, trust_remote_code=True)
    # datasets aquisition
    print("get datasets")
    train_dataset = get_dataset(os.path.join(args.data_dir, "train.csv"),
                                tokenizer=tokenizer, max_length= args.max_length)
    valid_dataset = get_dataset(os.path.join(args.data_dir, "valid.csv"),
                                tokenizer=tokenizer, max_length=args.max_length)
    # collate function
    print("set collate function")
    collate_fn = DataCollatorForSeq2Seq(tokenizer, pad_to_multiples_of=8,
                                        return_tensors="pt", padding=True)
    # get model
    print("get model")
    model = get_finetuning_model(model_cls, args.model_type, args.model_dir,
                                 lora = args.lora, lora_aplha = args.lora_alpha,
                                 lora_r = args.lora_r)
    # training arguments
    print("setting training arguments...")
    training_args = TrainingArguments(
        output_dir = args.output_dir,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_steps = args.warmup_steps,
        num_train_epochs = args.epochs,
        learning_rate = args.learning_rate,
        fp16 = args.fp16,
        logging_steps = args.logging_steps,
        save_strategy = "steps",
        save_steps = args.save_steps,
        save_total_limit = 5,
        deepspeed = args.deepspeed_config,
        gradient_checkpointing = args.gradient_checkpointing,
        report_to=None
    )
    print("traing arguments set done, start set trainer")
    # do training
    trainer = Trainer(
        model=model,
        train_dataset = train_dataset,
        eval_dataset = valid_dataset,
        args = args,
        data_collator=collate_fn
    )
    print("trainer set done, start training")
    trainer.train()
    

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser("simple chat bot")
    # base arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model_dir", type=str, default="./")
    # seed
    parser.add_argument("--seed", type=int, default=-1)
    # training arguments
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--deepspeed_config", type=str, default="./ds_zero2_config.json")
    # reporting
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=600)
    # the following should not be contradictory with deepspeed config
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--model_type", type=str, default="chatglm2")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    # lora settings
    parser.add_argument("--lora", type=str, default="")
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    # about tokenizer
    parser.add_argument("--max_length", type=int, default=128)
    # this argument will be filled automatically
    parser.add_argument("--local_rank", type=int, default=0)
    args=parser.parse_args()
    print(args)

    # seed setting
    set_random_seed(seed=args.seed)
    # main finetuning step
    main(args)