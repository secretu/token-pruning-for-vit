from models.l0_module import L0ModuleForMAC,L0Module
import torch
from trainer.trainer_vit import CoFiTrainer
# from trainer.trainer import CoFiTrainer
import transformers
import os
from copy import deepcopy
import sys
from utils.utils import *
from transformers import DeiTFeatureExtractor
from transformers.models.vit.modeling_vit import ViTForImageClassification
from transformers import (HfArgumentParser, TrainingArguments,set_seed)
from dataset_util import build_dataset
from models.model_vit import PrunViTForImageClassification
from transformers import AutoConfig
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
    model_args.model_name_or_path = 'facebook/deit-small-patch16-224'

    # data_args.data_path = '/data/imaginenet/ILSVRC2012_data_set_transformed'
    # train_dataset,nb_classes = build_dataset(is_train=True, args=data_args,pre_transform=True)
    # eval_dataset, _ = build_dataset(is_train=False, args=data_args,pre_transform=True)
    train_dataset,nb_classes = build_dataset(is_train=True, args=data_args,pre_transform=False)
    data_args.data_path = '/data/imaginenet/ILSVRC2012_data_set_transformed'  #eval dataset transformed
    eval_dataset, _ = build_dataset(is_train=False, args=data_args,pre_transform=True)
    
    # disable layer gate 
    # additional_args.disable_layer_gate = True
    # additional_args.select_token_pruning_layer = [0,1,2,3,4,5,6,7,8,9]
    ################################################
    training_args.eval_steps = 300
    training_args.logging_steps = 100
    training_args.num_train_epochs = 30
    training_args.per_device_train_batch_size = 32
    training_args.per_device_eval_batch_size = 32
    training_args.do_train = True
    training_args.do_eval = True
    training_args._n_gpu = 1
    training_args.learning_rate=1.5e-5       # classification lr
    additional_args.reg_learning_rate = 1e-2   # loga lr
    additional_args.lambda_learning_rate = 1e-2     # lambda lr
    additional_args.target_sparsity = 0.5
    additional_args.start_sparsity = 0.1
    additional_args.lagrangian_warmup_epochs = 1
    additional_args.sparsity_epsilon = 0.001
    training_args.label_smoothing_factor = 0.1  #0.001 
    training_args.dataloader_num_workers = 10   
    additional_args.bin_num = 196
    
    if training_args.output_dir == None:
        training_args.output_dir= '/home/chengquan/ToP-prune_before_FFN/train_out_put_dir/new_revisied/run_test'
    
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    training_args.logging_dir = os.path.join(training_args.output_dir,'logdir')
    os.makedirs(training_args.logging_dir,exist_ok=True)
    
    
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
    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=nb_classes,
        # finetuning_task=t_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
  
    # model  & teacher model
    model = PrunViTForImageClassification(config=config,token_prune_loc=additional_args.prune_location)
    teacher_model = PrunViTForImageClassification(config=config)
    weight_path = '/home/chengquan/ToP-prune_before_FFN/pretrained_model/deit_small_patch16_224.pth'
    model.load_state_dict(torch.load(weight_path))

    teacher_model.load_state_dict(torch.load(weight_path))
    teacher_model.eval()

    # maybe need further change
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
            start_sparsity=additional_args.start_sparsity,
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
        
    # def collate_fn(examples):
    #     x = []
    #     y = []
    #     for item in examples:
    #         x.append(item[0])
    #         y.append(torch.tensor([item[1]]))
    #     pixel_values = torch.stack([i for i in x])
    #     labels = torch.stack([i for i in y])
    #     B, C, HW, HW = pixel_values.size()
    #     num_tokens = 196 + 2 # for 14 * 14
    #     attention_mask = torch.ones(B,num_tokens).to(torch.float64)
        
    #     return {
    #         'pixel_values': pixel_values,
    #         'labels': labels,
    #         'attention_mask':attention_mask
    #     }
    
    def collate_fn(examples):
        pixel_values, labels = zip(*examples)
        pixel_values = torch.stack(pixel_values)
        labels = torch.stack([torch.tensor([label]) for label in labels])
        attention_mask = torch.ones(pixel_values.size(0), 196 + 1).to(torch.float64)
        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'attention_mask': attention_mask
        }
    
    feature_extractor = DeiTFeatureExtractor.from_pretrained(model_args.model_name_or_path)

    
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