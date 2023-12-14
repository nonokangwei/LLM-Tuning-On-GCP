import argparse
import numpy as np
import glob
import json
import logging
import os

from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer
from transformers import set_seed

from datasets import load_from_disk
import torch
import evaluate
import nltk

from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from google.cloud import storage
from google.cloud import aiplatform

import time

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/flan-t5-large",
        help="Default output directory where all artifacts will be written."
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        required=True,
        help="train dataset directory"
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        required=True,
        help="test dataset directory"
    )
    parser.add_argument(
        "--model_output_dir",
        type=str,
        required=True,
        help="Default output directory where all artifacts will be written."
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        type=str,
        required=True,
        help="Default Tensorboard log directory."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=3,
        help="epoch number"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        required=True,
        help="Passed by Deepspeed by default"
    )
    args = parser.parse_args()

    return args

args = parse_args()


MODEL_ID = args.model_id #"google/flan-t5-xxl"   # Model id to use for training, in Hugging face
TRAIN_DATASET_PATH = args.train_dataset_path #"data/train" # Path to processed dataset
TEST_DATASET_PATH  = args.test_dataset_path #"data/eval"  # Path to processed dataset
EPOCHS = args.epoch
PER_DEVICE_TRAIN_BATCH_SIZE = args.batch_size # Batch size to use for training
PER_DEVICE_EVAL_BATCH_SIZE = args.batch_size  # Batch size to use for testing
GENERATION_MAX_LENTH = 129      # Maximum length to use for generation
GENERATION_NUM_BEAMS = 4        # Number of beams to use for generation
LR = 1e-4 # Learning rate to use for training
SEED = 42 # Seed use for training
DEEPSPEED_FILE = "configs/ds_flan_t5_z3_config_bf16.json"             # Path to deepspeed config file
GRADIENT_CHECKPOINTING = True                                         # Whether to use gradient checkpointing
BF_16 = True if torch.cuda.get_device_capability()[0] == 8 else False # Whether to use bf16

# avoid noisy neighbour #3514 https://github.com/deepset-ai/haystack/issues/3514
try:
    nltk.data.find("tokenizers/punkt")
except (OSError, LookupError):
    try:
        nltk.download("punkt")
    except FileExistsError:
        pass

output_directory = args.model_output_dir

# Metric
metric = evaluate.load("rouge")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

# set seed
set_seed(SEED)

# load dataset from disk and tokenizer
train_dataset = load_from_disk(TRAIN_DATASET_PATH)
eval_dataset = load_from_disk(TEST_DATASET_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID,
    use_cache=False if GRADIENT_CHECKPOINTING else True  # this is needed for gradient checkpointing
)

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
)

# Define compute metrics function
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=args.model_output_dir,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    predict_with_generate=True,
    generation_max_length=GENERATION_MAX_LENTH,
    generation_num_beams=GENERATION_NUM_BEAMS,
    fp16=False,  # T5 overflows with fp16
    bf16=BF_16,  # Use BF16 if available
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    deepspeed=DEEPSPEED_FILE,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    # tensorboard
    logging_dir=args.tensorboard_log_dir,#
    report_to=["tensorboard"],
    # logging & evaluation strategies
    logging_strategy="steps",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=10000,
    load_best_model_at_end=False,  # avoiding OOM errors

    # push to hub parameters
    push_to_hub=False, # no puhsing to hub
    hub_strategy="every_save",
    hub_model_id=None,
    hub_token=None
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

start_time = time.time()
# Start training and evaluation
logging.info("Training ....")
torch.cuda.empty_cache()
trainer.train()
logging.info("Evaluating ....")
metrics = trainer.evaluate()

end_time = time.time()
duration = end_time - start_time
print("training duration is " + str(duration))
