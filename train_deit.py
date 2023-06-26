from models.l0_module import L0ModuleForMAC,L0Module
from pathlib import Path
from datasets import load_dataset, load_metric, DatasetDict
import torch
from trainer.trainer import CoFiTrainer 
import transformers
import torch.backends.cudnn as cudnn
import os
from copy import deepcopy
import numpy as np
import sys
from utils.utils import *
from transformers import DeiTFeatureExtractor
from transformers import (HfArgumentParser, TrainingArguments, PretrainedConfig,
                          glue_output_modes, glue_tasks_num_labels, set_seed)
from sklearn.metrics import accuracy_score,top_k_accuracy_score
from dataset_util import build_dataset, build_transform
from transformers.models.deit.modeling_deit import DeiTConfig
from models.model_deit import PrunDeiTModelForClassficationWithTeacher
from transformers import AutoConfig, AutoTokenizer, EvalPrediction, default_data_collator, DataCollatorWithPadding
from args import AdditionalArguments, DataTrainingArguments
from models.model_args import ModelArguments
from utils.cofi_utils import *

import logging
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")


def main():    
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalArguments))
    model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()

    training_args.eval_steps = 100
    training_args.logging_steps = 20
    training_args.num_train_epochs = 10
    training_args.per_device_train_batch_size = 32
    training_args.per_device_eval_batch_size = 32
    training_args.do_train = True
    training_args.do_eval = True
    training_args._n_gpu = 4
    training_args.learning_rate=1.5e-05
    additional_args.reg_learning_rate = 1e-3
    additional_args.target_sparsity = 0.5
    additional_args.lagrangian_warmup_epochs = 5
    training_args.label_smoothing_factor = 0.1  #0.001 
    
    
    
    if training_args.output_dir == None:
        training_args.output_dir= '/home/chengquan/ToP-prune_before_FFN/train_out_put_dir/test_run_3'
    os.makedirs(training_args.output_dir, exist_ok=True)
    assert additional_args.prune_location == [2,3,4,5,6,7,8,9,10,11], "only support prune location 2,3,4,5,6,7,8,9,10,11"
   

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # save args
    torch.save(training_args, os.path.join(
        training_args.output_dir, "training_args.bin"))
    torch.save(data_args, os.path.join(
        training_args.output_dir, "data_args.bin"))
    torch.save(model_args, os.path.join(
        training_args.output_dir, "model_args.bin"))
    torch.save(additional_args, os.path.join(
        training_args.output_dir, "additional_args.bin"))
 

    set_seed(training_args.seed)

    # print all arguments
    # log_all_parameters(logger, model_args, data_args,
    #                    training_args, additional_args)
    train_dataset,nb_classes = build_dataset(is_train=True, args=data_args)
    eval_dataset, _ = build_dataset(is_train=False, args=data_args)
    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=nb_classes,
        # finetuning_task=t_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # print(config)
    # quit()
    # config = DeiTConfig()
    
    model = PrunDeiTModelForClassficationWithTeacher(config=config,token_prune_loc=additional_args.prune_location)
    
    weight_path = '/home/chengquan/ToP-prune_before_FFN/pretrained_model/deit-small-distilled-patch16-224'
    model.load_state_dict(torch.load(weight_path))
    # quit()
    # maybe need further change
    teacher_model = None
    if additional_args.do_distill:
        teacher_model = model.from_pretrained(
            additional_args.distillation_path,
            config=deepcopy(config),
        )
        teacher_model.eval() #! inside has a cofibertmodel #! CofiBertForSequenceClassification
    config.do_layer_distill = additional_args.do_layer_distill #! True
    if additional_args.do_layer_distill:
        initialize_layer_transformation(model)
    

    logger.info(model)
    logger.info(f"Model size: {calculate_parameters(model)}")

    l0_module = None
    if additional_args.pruning_type is not None:
        l0_module_class = L0ModuleForMAC if additional_args.use_mac_l0 else L0Module
        l0_module = l0_module_class(
            config=config,
            droprate_init=additional_args.droprate_init,
            temperature=additional_args.temperature,
            target_sparsity=additional_args.target_sparsity,
            pruning_type=additional_args.pruning_type,
            token_prune_loc=additional_args.prune_location,
            bin_num=additional_args.bin_num,
            disable_layer_gate=additional_args.disable_layer_gate,
            include_padding=additional_args.include_padding,
            select_token_pruning_layer=additional_args.select_token_pruning_layer,
        ).cuda()

    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = int(target.size()[0])
            
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(float(correct_k.mul_(100.0 / batch_size))) 
            return res
        
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        pred = torch.from_numpy(predictions)
        
        target = torch.from_numpy(labels)
        acc1, acc5 = accuracy(pred, target, topk=(1, 5))
        return {"top1ACC": acc1, "top5ACC": acc5}
        # return {"top1ACC": acc1}
        
    def collate_fn(examples):
        x = []
        y = []
        for item in examples:
            x.append(item[0])
            y.append(torch.tensor([item[1]]))
        pixel_values = torch.stack([i for i in x])
        labels = torch.stack([i for i in y])
        B, C, HW, HW = pixel_values.size()
        num_tokens = 196 + 2 # for 14 * 14
        attention_mask = torch.ones(B,num_tokens).to(torch.float64)
        
        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'attention_mask':attention_mask
        }
    
    feature_extractor = DeiTFeatureExtractor.from_pretrained(model_args.model_name_or_path)

    
    
    teacher_model = PrunDeiTModelForClassficationWithTeacher(config=config,token_prune_loc=None)
    
    # weight_path = '/home/chengquan/ToP-prune_before_FFN/pretrained_model/deit-small-distilled-patch16-224'
    teacher_model.load_state_dict(torch.load(weight_path))
    teacher_model = None
    
    trainer = CoFiTrainer(
        model=model,
        args=training_args,
        additional_args=additional_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        data_collator=collate_fn,
        l0_module=l0_module,
        teacher_model=teacher_model,
    )
    from transformers.integrations import AzureMLCallback, ProgressCallback
    trainer.remove_callback(AzureMLCallback)
    trainer.remove_callback(ProgressCallback)


    print("Training Arguments")
    print(training_args)
    print("Additional Arguments")
    print(additional_args)
    # trainer.evaluate()
    
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # tokenizer.save_pretrained(training_args.output_dir)









if __name__ == '__main__':
    import time
    os.environ["WANDB_DISABLED"] = "true"
    t_start = time.time()
    main()
    t_end = time.time()
    logger.info(f"Training took {round(t_end - t_start, 2)} seconds.")