from typing import Dict, List, Any
import torch
from transformers import (
    PreTrainedTokenizerFast,
    AutoModelForSeq2SeqLM,
)

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global tokenizer
    global device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSeq2SeqLM.from_pretrained("heooo/kobartTest")
    model = model.to(device)

    tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer
    global device

    # Parse out your arguments
    dataPop = model_inputs.get('inputs', None)

    if dataPop == None:
        return {'message': "No prompt provided"}

    if isinstance(dataPop, str):
        texts = [dataPop]
    else:
        texts = dataPop

    #parmeters
    beam = 5
    sampling =  False
    temperature = 1.0
    sampling_topk = -1
    sampling_topp = -1
    length_penalty = 1.0
    max_len_a = 1
    max_len_b = 50
    no_repeat_ngram_size = 4
    return_tokens = False
    bad_words_ids = None

    tokenized = tokenize(texts)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    generated = model.generate(
        input_ids.to(device),
        attention_mask=attention_mask.to(device),
        use_cache=True,
        early_stopping=False,
        decoder_start_token_id=tokenizer.bos_token_id,
        num_beams=beam,
        do_sample=sampling,
        temperature=temperature,
        top_k=sampling_topk if sampling_topk > 0 else None,
        top_p=sampling_topp if sampling_topk > 0 else None,
        no_repeat_ngram_size=no_repeat_ngram_size,
        bad_words_ids=[[tokenizer.convert_tokens_to_ids("<unk>")]]
        if not bad_words_ids else bad_words_ids +
        [[tokenizer.convert_tokens_to_ids("<unk>")]],
        length_penalty=length_penalty,
        max_length=max_len_a * len(input_ids[0]) + max_len_b,
    )

    summ_result = ''
    if return_tokens:
        output = [
            tokenizer.convert_ids_to_tokens(_)
            for _ in generated.tolist()
        ]

        summ_result = (output[0] if isinstance(
            dataPop,
            str,
        ) else output)

    else:
        output = tokenizer.batch_decode(
            generated.tolist(),
            skip_special_tokens=True,
        )

        summ_result = (output[0].strip() if isinstance(
            dataPop,
            str,
        ) else [o.strip() for o in output])
        
    return {"summarization": summ_result}

def tokenize(texts: List[str], max_len: int = 1024) -> Dict:
    global tokenizer

    if isinstance(texts, str):
        texts = [texts]

    texts = [f"<s> {text}" for text in texts]
    eos = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    eos_list = [eos for _ in range(len(texts))]

    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
        max_length=max_len - 1,
        # result + <eos>
    )

    return add_bos_eos_tokens(tokenizer, tokens, eos_list)

def add_bos_eos_tokens(tokenizer, tokens, eos_list):
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    token_added_ids, token_added_masks = [], []

    for input_id, atn_mask, eos in zip(
            input_ids,
            attention_mask,
            eos_list,
    ):
        maximum_idx = [
            i for i, val in enumerate(input_id)
            if val != tokenizer.convert_tokens_to_ids("<pad>")
        ]

        if len(maximum_idx) == 0:
            idx_to_add = 0
        else:
            idx_to_add = max(maximum_idx) + 1

        eos = torch.tensor([eos], requires_grad=False)
        additional_atn_mask = torch.tensor([1], requires_grad=False)

        input_id = torch.cat([
            input_id[:idx_to_add],
            eos,
            input_id[idx_to_add:],
        ]).long()

        atn_mask = torch.cat([
            atn_mask[:idx_to_add],
            additional_atn_mask,
            atn_mask[idx_to_add:],
        ]).long()

        token_added_ids.append(input_id.unsqueeze(0))
        token_added_masks.append(atn_mask.unsqueeze(0))

    tokens["input_ids"] = torch.cat(token_added_ids, dim=0)
    tokens["attention_mask"] = torch.cat(token_added_masks, dim=0)
    return tokens
