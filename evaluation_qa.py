import inspect
import os
import pdb
import random
import sys
import time

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers.data.data_collator import (DataCollator,
                                             DataCollatorWithPadding,
                                             default_data_collator)
from transformers.trainer_pt_utils import nested_concat, nested_numpify
from transformers.trainer_utils import EvalPrediction

from models.modeling_bert import (CoFiBertForQuestionAnswering,
                                  CoFiBertForSequenceClassification)
from models.modeling_roberta import CoFiRobertaForSequenceClassification
from utils.cofi_utils import *
from utils.qa_utils import *
from utils.utils import *
from thop import profile, clever_format
from models.l0_module import L0ModuleForMAC
from transformers import AutoConfig
config = AutoConfig.from_pretrained("bert-base-uncased")
prune_location = list(map(int, sys.argv[3].split(","))) if len(sys.argv) >= 4 else []
l0_module = L0ModuleForMAC(config, pruning_type="token+pruner", token_prune_loc=prune_location)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def _remove_unused_columns(dataset: "datasets.Dataset", description):
    # Inspect model forward signature to keep only the arguments it accepts.
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    # Labels may be named label or label_ids, the default data collator handles that.
    signature_columns += ["label", "label_ids"]
    columns = [k for k in signature_columns if k in dataset.column_names]
    ignored_columns = list(set(dataset.column_names) - set(signature_columns))
    dset_description = "" if description is None else f"in the {description} set "
    print(
        f"The following columns {dset_description} don't have a corresponding argument in `{model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
    )
    dataset.set_format(type=dataset.format["type"], columns=columns)


def get_dataloader(dataset, batch_size):
    dataloader = DataLoader(dataset,
                            sampler=SequentialSampler(dataset),
                            batch_size=batch_size,
                            collate_fn=default_data_collator)
    return dataloader

def post_processing_function(examples, features, predictions):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v}
                             for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex[answer_column_name]}
                  for ex in datasets["validation"]]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def shortens_inputs(inputs):
    max_length = inputs["attention_mask"].sum(-1).max().item()
    inputs["input_ids"] = inputs["input_ids"][:, :max_length]
    inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
    if "token_type_ids" in inputs:
        inputs["token_type_ids"] = inputs["token_type_ids"][:, :max_length]


def calculate_flops(statistics):
    baseline_effective_lengths = np.array(statistics["baseline_effective_lengths"]).T
    pruned_effective_lengths = np.array(statistics["pruned_effective_lengths"]).T
    baseline_flops = []
    pruned_flops = []
    for baseline_effective_length, pruned_effective_length in zip(baseline_effective_lengths, pruned_effective_lengths):
        baseline_flop = l0_module.calculate_mac_for_model(token_length_for_each_layer=baseline_effective_length)
        pruned_flop = l0_module.calculate_mac_for_model(token_length_for_each_layer=pruned_effective_length)
        baseline_flops.append(baseline_flop)
        pruned_flops.append(pruned_flop)
    return baseline_flops, pruned_flops


def analyze_results(logits, inputs, model, tokenizer):
    pred = logits.argmax(-1)
    labels = inputs["labels"]
    correct = pred.eq(labels)
    correct = correct.cpu().numpy()
    input_ids = inputs["input_ids"]
    token_lengths = inputs['attention_mask'].sum(-1)

    print(correct)

    for P, pred_score in enumerate(model.bert.encoder.pred_scores):
        last_pred_score = pred_score[..., 0]
        if model.bert.encoder.masks[P] is not None:
            last_pred_score *= model.bert.encoder.masks[P]
        score_rank = last_pred_score.argsort(-1, descending=True)
        for i, (rank, ids, token_length, cor) in enumerate(zip(score_rank, input_ids, token_lengths, correct)):
            if i != 30:
                continue
            sentence = tokenizer.convert_ids_to_tokens(ids[:token_length])
            if P == 0:
                print(labels[P].item(), sentence)
            seq_token_score = last_pred_score[i][(ids == 102)[1:]]
            rank = rank[:int(token_length * 0.3)] + 1
            important_ids = ids[rank]
            # convert token from id to word
            important_words = tokenizer.convert_ids_to_tokens(important_ids)
            print(P, cor, important_words)
            print(P, cor, [round(l, 4) for l in last_pred_score[i][rank - 1].cpu().numpy().tolist()], seq_token_score)
            break
    exit(0)
    print("analyze result...")


def evaluate(model, zs, tokenizer):
    metrics = {}
    total_examples = 0
    _remove_unused_columns(dataset, "evaluation")

    preds = None
    label_ids = None
    total_infer_time = 0
    baseline_flops = []
    pruned_flops = []


    csv_data = []
    input_sequence_index = 0
    for num_batch, inputs in enumerate(dataloader):


        if num_batch < 1:
            continue

        relative_macs = l0_module.get_mac_and_constraint(
            attention_mask=inputs["attention_mask"],
            token_score=zs["token_z"] if zs is not None else None,
            pruner_score=zs["pruner_z"] if zs is not None else None,
        )
        labels = inputs["labels"] if "labels" in inputs else None
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        with torch.no_grad():
            a = time.time()
            if task_name == "squad_v2":
                output = model(**inputs)
                logits = output["start_logits"], output["end_logits"]
            else:
                logits = model(**inputs)["logits"]
            torch.cuda.synchronize()

            input_ids = inputs["input_ids"][0][1:]
            valid_tokens_num = inputs["attention_mask"][0][1:].sum()
            scores = model.bert.encoder.pred_scores[1][..., 0][0]
            last_scores = model.bert.encoder.pred_scores[-1][..., 0][0]
            # convert id to word
            sentence = tokenizer.convert_ids_to_tokens(input_ids)
            sentence = sentence[:valid_tokens_num]
            scores = scores[:valid_tokens_num].cpu().numpy().tolist()
            scores = [round(score, 4) for score in scores]
            last_scores = last_scores[:valid_tokens_num].cpu().numpy().tolist()
            last_scores = [round(score, 4) for score in last_scores]
            data = [(score, word) for score, word in zip(scores, sentence)]
            print("data:", data)
            data = sorted(data, key=lambda x: x[0], reverse=True)
            last_data = [(score, word) for score, word in zip(last_scores, sentence)]
            print("last data:", last_data)
            last_data = sorted(last_data, key=lambda x: x[0], reverse=True)
            from pprint import pprint
            print(data)
            print(" ".join(sentence))
            print(num_batch)
            for i in range(len(data)):
                if data[i][1] == "10th":
                    print(i, data[i])
            print("-----------------")
            for i in range(len(last_data)):
                if last_data[i][1] == "10th":
                    print(i, last_data[i])
            print(len(scores))
            import pdb; pdb.set_trace()
            continue

            # analyze_results(logits, inputs, model, tokenizer)
            lengths = inputs["attention_mask"].sum(-1) - 1
            for i in range(len(labels)):
                for layer in range(12):
                    score = model.bert.encoder.pred_scores[layer]
                    for token_index, token_score in enumerate(score[i]):
                        if token_index == lengths[i]:
                            break
                        csv_data.append([input_sequence_index, layer, token_index, token_score.item()])
                input_sequence_index += 1

            b = time.time()
            total_infer_time += (b-a)
            total_examples += len(logits)
            preds = logits if preds is None else nested_concat(
                preds, logits)
            label_ids = labels if label_ids is None else nested_concat(
                label_ids, labels)
            if zs is not None:
                batch_baseline_flops, batch_pruned_flops = calculate_flops(model.bert.encoder.inference_statistics)
                baseline_flops.extend(batch_baseline_flops)
                pruned_flops.extend(batch_pruned_flops)
            

            # import csv
            # f = open("data.csv", "w")
            # writer = csv.writer(f)
            # # write csv header
            # writer.writerow(["input_sequence_index", "layer", "token_index", "score"])
            # writer.writerows(csv_data)
            # f.close()


    if label_ids is not None:
        final_label_ids = nested_numpify(label_ids)
    if preds is not None:
        final_preds = nested_numpify(preds)

    metrics["num_examples"] = total_examples
    if task_name == 'squad_v2':
        dataset.set_format(
            type=dataset.format["type"], columns=list(dataset.features.keys()))
        eval_preds = post_processing_function(
            eval_examples, dataset, final_preds)
        metrics = compute_metrics(eval_preds)
    else:
        metrics = compute_metrics(EvalPrediction(
            predictions=final_preds, label_ids=final_label_ids))
    metrics["seconds/example"] = total_infer_time / total_examples
    metrics["baseline_flops"] = np.mean(baseline_flops)
    metrics["pruned_flops"] = np.mean(pruned_flops)
    metrics["sparsity"] = (metrics["baseline_flops"] - metrics["pruned_flops"]) / metrics["baseline_flops"]
    return metrics


def prepare_validation_features(examples):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    max_length = 384
    doc_stride = 128
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation.py, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def glue_preprocess_function(examples):
    # Tokenize the texts
    sentence1_key, sentence2_key = task_to_keys[task_name]
    max_seq_length = 128
    padding = "max_length"
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (
            examples[sentence1_key], examples[sentence2_key])
    )

    result = tokenizer(*args, padding=padding,
                       max_length=max_seq_length, truncation=True)
    if task_name == "mnli" and model_name_or_path.startswith("princeton-nlp/"):
        # legacy issue of using GLUEDataset
        label_to_id = {1:2, 0:1, 2:0}
        labels = [label_to_id[i] for i in examples["label"]]
        result["label"] = labels
    return result


def warmup():
    time1 = time.time()
    input = torch.randn(128, 1024).cuda()
    linear = torch.nn.Linear(1024, 1024).cuda()
    for i in range(10000):
        input = linear(input)

    time2 = time.time()
    print(round(time2 - time1, 2), "seconds for warmup")

def get_glue_metric():
    metric = load_metric("glue", task_name)
    is_regression = task_name == "stsb"

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    return compute_metrics

if __name__ == '__main__':
    # warmup
    warmup()

    # data
    task_name = sys.argv[1].lower()
    model_name_or_path = sys.argv[2]

    bs = 4

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True if task_name == "squad_v2" else False, padding_side="right", truncation_size="right")

    if task_name != "squad_v2":
        # data_args = DataTrainingArguments(task_name=task_name,
        #   data_dir=os.path.join(data_dir, task_name))
        # dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")
        if task_name == "mnli":
            set_name = "validation_matched"
        else:
            set_name = "validation"
        dataset = datasets.load_dataset("glue", task_name)[set_name]
        dataset = dataset.map(glue_preprocess_function, batched=True)

        compute_metrics = get_glue_metric()
    else:
        metric = load_metric("squad_v2")

        def compute_metrics(p: EvalPrediction):
            return metric.compute(predictions=p.predictions, references=p.label_ids)

        datasets = load_dataset("squad_v2")
        column_names = datasets["validation"].column_names
        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        # Padding side determines if we do (question|context) or (context|question).
        pad_on_right = tokenizer.padding_side == "right"
        dataset = datasets["validation"].map(
            prepare_validation_features,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
        )
        eval_examples = datasets["validation"]

    dataloader = get_dataloader(dataset, bs)

    # load model
    if "squad_v2" in task_name:
        model_class = CoFiBertForQuestionAnswering
    else:
        model_class = CoFiBertForSequenceClassification

    zs = load_zs(model_name_or_path)

    # for compressed models
    if zs is None:
        model = model_class.from_pretrained(model_name_or_path, token_prune_loc=prune_location)
    # for full models with compression vectors zs
    else:
        model = load_model(model_name_or_path, model_class, zs, token_prune_loc=prune_location)

    model = model.cuda()
    model = model.eval()

    model.config.output_hidden_states = False
    model.config.output_attentions = False

    metrics = evaluate(model, zs, tokenizer)
    print(f"Task: {task_name}")
    print(f"Model path: {model_name_or_path}")
    print(f"Sparsity: {metrics['sparsity']}")
    for key in metrics:
        if key in ["macs"]:
            continue
        print(f"{key}: {round(metrics[key], 6 if 'seconds' in key else 4)}")
    print()
