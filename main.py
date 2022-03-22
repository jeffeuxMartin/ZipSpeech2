#!python
import sys, pathlib, argparse
import torch
from transformers import (
    AutoTokenizer, Trainer, TrainingArguments)
# from datasets import load_metric
from src.mydatasets import MyDataset
from src.mymodel import BartHuBertAutoEncoder
from src.utils import (
    compute_metrics, range_checker, unpad_sequence)
from src.utils import get_args

# region --- defs
# endregion

args = get_args()
PREPADDING_ID = -1
MAX_LEN_MY = args.max_len

device = ('cpu', 'cuda')[torch.cuda.is_available()]
filedir = pathlib.Path(__file__).parent.resolve()

# region --- load codes
stacked_codes = torch.load(
    # "/storage/LabJob/DiscreteCodeRevive/"
    filedir /
    "stacked_codes.tnsr.pt")

*code_range, prepadding = range_checker(
    stacked_codes, PREPADDING_ID=PREPADDING_ID)

if code_range[0] == 0:
    # idx 0 is used --> special token in tail!
    BOS_TOKEN = code_range[1] + 1
    EOS_TOKEN = code_range[1] + 2
    UNK_TOKEN = code_range[1] + 3
    PAD_TOKEN = code_range[1] + 4
else:
    raise NotImplementedError
    
seqs = unpad_sequence(
    stacked_codes,
    (stacked_codes != PREPADDING_ID).sum(-1),
    batch_first=True,
)
# endregion
# region --- load texts
SEPARATOR = '\n# @@@ === @@@ #\n'
with open(
    # "/storage/LabJob/DiscreteCodeRevive/"
    filedir /
    'stacked_texts.txt') as f:
    stacked_texts = f.read().split(SEPARATOR)

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/bart-base")

word_lengths = [sum(i) for i in (
    tokenizer.batch_encode_plus(
        stacked_texts)['attention_mask'])]  
        # just keep lengths
# endregion
# ~~~ # ~~~ # ~~~ # ~~~ # ~~~ # ~~~ # ~~~ # ~~~ # ~~~ # ~~~ # ~~~ # ~~~ 
    
# seqs = seqs[:200]                   # XXX: Just for test
# word_lengths = word_lengths[:200]   # XXX: Just for test
# TODO: KAIWEI_SHUT_THE_FUCK_UP: 長度排序了！要打亂～
# TODO: KAIWEI_SHUT_THE_FUCK_UP: 真的有按照每個 batch 去重新 pad 嗎？
# SPLITPOINT = int(round(len(seqs) * 0.8))
# train_dataset = MyDataset(
#     seqs[:SPLITPOINT], 
#     word_lengths[:SPLITPOINT],
#     PAD_TOKEN, MAX_LEN_MY)
# valid_dataset = MyDataset(
#     seqs[SPLITPOINT:], 
#     word_lengths[SPLITPOINT:],
#     PAD_TOKEN, MAX_LEN_MY)
VAL_PART = args.validation
train_dataset = MyDataset(
    [seqs[idx] 
     for idx in range(len(seqs)) 
     if idx % VAL_PART != VAL_PART - 1], 
    [word_lengths[idx] 
     for idx in range(len(word_lengths)) 
     if idx % VAL_PART != VAL_PART - 1], 
    PAD_TOKEN, MAX_LEN_MY)
valid_dataset = MyDataset(
    [seqs[idx] 
     for idx in range(len(seqs)) 
     if idx % VAL_PART == VAL_PART - 1], 
    [word_lengths[idx] 
     for idx in range(len(word_lengths)) 
     if idx % VAL_PART == VAL_PART - 1], 
    PAD_TOKEN, MAX_LEN_MY)

model = BartHuBertAutoEncoder.from_pretrained(
    "facebook/bart-base",
    PAD_TOKEN=PAD_TOKEN,
).to(device)

training_args = TrainingArguments(
    output_dir=args.output_dir,

    do_train=True,
    logging_steps=10,
    per_device_train_batch_size=args.batch_size_train,

    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=10 * 18,
    per_device_eval_batch_size=args.batch_size_eval,

    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    warmup_steps=args.warmup_steps,
    report_to="wandb",
    optim="adamw_torch",
    label_names=[
        "input_ids", 
        "attention_mask",
    ],
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=train_dataset.collator,  # \
                                 # May be problemsome!
    # tokenizer=model.tokenizer,
    # callbacks=[
    #     EarlyStoppingCallback(
    #         early_stopping_patience=20)],
)

trainer.train(
    ignore_keys_for_eval=[
        "last_hidden_state",
        "encoder_last_hidden_state",
        "acc",
    ] + (trainer.model.config
        .keys_to_ignore_at_inference),
)











