# Simple Chatbot
A pytorch simple chatbot framework for training and inference. Such frame work is far from perfect and only support T5 and ChatGLM2. We warmly welcome you to contribute to this repo for better and easier using, support of more backbones and RLHF in the future. Interactive functions are also welcomed.

# Environments
I have tried it works torch2.0 with the following environment, requirement.txt will be developed later.
```
pandas
numpy==1.25
deepspeed
accelerate
transformers==4.30.2
```

# Dataset
The current version only supports .csv training files, wheras many training data are in .jsonl format. We provide a simple code example for you to convert your raw json lines to a csv file.
``` python
import pandas as pd
import json

def load_train_data(data_path):
    preline = lambda x: json.loads(x.strip())
    with open(data_path, 'r', encoding='utf-8') as f:
        train_data = [preline(line) for line in f]
    df_train = pd.DataFrame(train_data)
    return df_train

if __name__ == "__main__":
    dft = load_train_data("data/train.jsonl")
    dft.to_csv("data/train.csv")
```

# Training
A training command examples are shown as following (adalora, zero2 and gradient checkpointing are adopted)
``` Bash
deepspeed --num_gpus=2 finetune.py --model_dir=$YOUR_PRETRAINED_MODEL_PATH --fp16 --batch_size=2 --max_length=3900 --save_steps=1000 --epochs=2 --warmup_steps=200 --gradient_accumulation_steps=2 --lora=adalora -—gradient_checkpointing
```
You can set the location at which the checkpoints save, the default location is `./checkpoints`, please refer to finetune.py for more details.

# Inference
After you save you model at some path, inference is quite easy. Just run the following:
``` Bash
python inference.py --file_path=$YOUR_TEST_DATA_FILE --lora_ckpt=$YOUR_LORA_WEIGHT_SAVE_DIR --model_dir=$YOUR_BASE_OR_TRAINED_MODEL_DIR --max_new_tokens=256 --max_length=3900 --input_key=$YOUR_INPUT_SENTENCE_KEY --output_dir=result_2000.csv
```
where the `input_key` is the column at which your input sentences are. Note that there is an exisiting issue: Since I adopt `Huggingface Trainer` which only save model weights rather than both the model and the tokenizer. This give rise to some errors you might encounter while conducting inference. So once you have done your full paramter SFT, it is recommended to do:
``` Bash
cp $YOUR_PRETRAINED_MODEL_PATH/tokenizer_config.json $YOUR_TRAINED_MODEL_DIR
cp $YOUR_PRETRAINED_MODEL_PATH/tokenizer.model $YOUR_TRAINED_MODEL_DIR
```
I will manage to tackle such problem soon later. And it works ok while you set your training mode to "adalora".

# Contribution
Please contact johnnyrichards@163.com, and we welcome you to do further contribution.