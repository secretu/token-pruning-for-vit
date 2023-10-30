from transformers.models.deit.modeling_deit import (
    DeiTConfig,DeiTEmbeddings,DeiTSelfAttention,DeiTForImageClassificationWithTeacherOutput,
    DeiTAttention,DeiTIntermediate,DeiTLayer,DeiTEncoder,DeiTSelfOutput,
    DeiTModel,DeiTPooler,BaseModelOutputWithPooling,DeiTPreTrainedModel,DeiTOutput)
import torch
import math
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Set, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import os
from transformers import AutoConfig
from transformers.models.vit.modeling_vit import ViTEmbeddings,ViTSelfOutput,ViTIntermediate,ViTOutput,ViTPooler,ViTPreTrainedModel


class PrunDeiTLayerNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input, hidden_z=None):
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            compressed_input = torch.index_select(
                input, dim=-1, index=remaining_index)
            compressed_weight = self.weight[remaining_index]
            compressed_bias = self.bias[remaining_index]
            normalized_shape = len(remaining_index)
            normed_input = F.layer_norm(
                compressed_input, [normalized_shape], compressed_weight, compressed_bias, self.eps)
            output = input.clone()
            output[:, :, remaining_index] = normed_input
        else:
            output = F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)
        return output


class PrunDeiTSelfAttention(DeiTSelfAttention):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__(config)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        self.token_score = None
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)


        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    def transpose_for_scores(self, x):
        x_shape = x.size()
        last_dim = x_shape[-1]
        size_per_head = last_dim // self.num_attention_heads
        new_x_shape = x_shape[:-1] + (self.num_attention_heads, size_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) 
    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)
    
    def store_token_score(self, attention_probs, value_layer):
        # from arxiv 2111.15667
        V_norm = torch.norm(value_layer, dim=-1)
        # the first row represents the importance of the input token j for the output classification token
        A = attention_probs[:, :, 0, :]
        score_vectors = A * V_norm
        # skip the first [CLS] token
        score_vectors = score_vectors[..., 1:]
        score_vectors /= score_vectors.sum(dim=-1, keepdim=True)
        # sum token score over all heads
        token_score = score_vectors.sum(dim=1)
        self.token_score = token_score
    def forward(self,
                hidden_states,
                attention_mask=None,
                output_attentions=False,
                head_mask=None,
                head_z=None,
                policy=None):
        if self.value is None:
            return (None, None) if output_attentions else (None,)

        query_hidden_states = hidden_states
        mixed_query_layer = self.query(query_hidden_states)

        key_hidden_states = hidden_states
        mixed_key_layer = self.key(key_hidden_states)

        value_hidden_states = hidden_states
        mixed_value_layer = self.value(value_hidden_states)

        batch_size, seq_length, _ = hidden_states.shape

        if not hasattr(self, "ones"):
            self.ones = torch.ones(batch_size, seq_length, seq_length).float().to(
                hidden_states.device)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        
        attention_mask =attention_mask.to(torch.float32)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        if policy is None:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
        else:
            raise NotImplementedError
            attention_probs = self.softmax_with_policy(attention_scores, policy)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        value_layer = self.transpose_for_scores(mixed_value_layer)
        self.store_token_score(attention_probs, value_layer)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        if head_z is not None:
            context_layer *= head_z

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size(
        # )[:-2] + (context_layer.shape[-1] * context_layer.shape[-2],)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs,) if output_attentions else (
            context_layer,)
        return outputs
    

class PrunDeiTAttention(DeiTAttention):
    def __init__(self, config):
        super().__init__(config)
        self.attention = PrunDeiTSelfAttention(config)
        self.output = DeiTSelfOutput(config)
        self.config = config
        self.pruned_heads = set()
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        head_mask= None,
        head_z=None,
        policy=None,
    ):
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions,
            head_mask=head_mask,
            head_z=head_z,
            policy=policy,
        )

        attention_output = self.output(
            self_outputs[0],hidden_states)
        outputs = (attention_output,) + self_outputs
        return outputs
    
class PrunDeiTLayer(DeiTLayer):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__(config)
        self.attention = PrunDeiTAttention(config)
        
        self.intermediate = DeiTIntermediate(config)
        self.output = DeiTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        head_mask= None,
        output_attentions=False,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
        policy=None,
    ):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            head_z=head_z,
            # head_layer_z=head_layer_z,
            policy=policy,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states
        
        # in DeiT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        if intermediate_z is not None:
            layer_output = layer_output.mul(intermediate_z)
       
        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)
        
        outputs = (layer_output,) + outputs + (attention_output, )
        
        return outputs

class PrunDeiTEncoder(DeiTEncoder):
    def __init__(self, config, token_prune_loc=None):
        super().__init__(config)
        self.layer = nn.ModuleList([PrunDeiTLayer(config)
                                   for _ in range(config.num_hidden_layers)])
        
        if token_prune_loc is None:
            token_prune_loc = []
            print("disable token pruning.")
        else:
            print("enable token pruning. token_prune_loc: {}".format(token_prune_loc))
        self.include_padding = False
        self.token_prune_loc = token_prune_loc
        self.hard_token_mask = None
        self.hard_pruner_mask = None
        self.masks = []
        self.pred_scores = []
        self.inference_statistics = dict(
            baseline_effective_lengths=[],
            pruned_effective_lengths=[],
        )
        for _ in range(config.num_hidden_layers):
            self.masks.append(None)
            self.pred_scores.append(None)
            self.inference_statistics["baseline_effective_lengths"].append(None)
            self.inference_statistics["pruned_effective_lengths"].append(None)

    def get_hard_keep_decision_for_training(
        self,
        pred_score: torch.Tensor,
        rank_mask: torch.Tensor,
        prev_decision: torch.Tensor,
        attention_mask: torch.Tensor,
        pruner_mask: torch.Tensor,
        # excluded_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        binary_attention_mask = (attention_mask > -1).float()
        binary_prev_decision = (prev_decision > 0.0).float()
        token_importance_score = (pred_score[..., 0] * binary_prev_decision.squeeze(-1)).detach()
        token_index = torch.argsort(token_importance_score, dim=1, descending=True)
        token_rank = torch.argsort(token_index, dim=1)
        effective_token_length = torch.sum(binary_prev_decision.squeeze(-1), dim=1).long()
        if self.include_padding:
            effective_token_length = torch.ones_like(effective_token_length, device=effective_token_length.device) * token_importance_score.size(1)
        token_rank = ((token_rank / (effective_token_length + 1e-6).unsqueeze(-1)).clamp(0.0, 1.0) * len(rank_mask)).long().clamp_min(0)
        rank_mask_with_padding = torch.hstack([rank_mask, torch.tensor(0.0, device=rank_mask.device, dtype=rank_mask.dtype)])
        soft_rank_keep_mask = rank_mask_with_padding[token_rank]
        hard_keep_decision = soft_rank_keep_mask
        hard_keep_decision = hard_keep_decision.unsqueeze(-1)
        binary_pruner_mask = (pruner_mask > 0.0).float()
        hard_pruner_mask = (binary_pruner_mask - pruner_mask).detach() + pruner_mask
        hard_keep_decision = (1.0 - ((1.0 - hard_keep_decision) * hard_pruner_mask)) * binary_prev_decision
        # hard_keep_decision = torch.where(
        #     excluded_token_mask[..., 1:].unsqueeze(-1),
        #     torch.ones_like(hard_keep_decision),
        #     hard_keep_decision,
        # )
        return hard_keep_decision

    def get_new_attention_mask_for_inference(
        self,
        token_score, rank_mask, attention_mask, pruner_mask, #excluded_token_mask,
    ):
        # if pruner mask is zero, it means that the pruner is not used in this layer.
        if pruner_mask == 0.0:
            return attention_mask
        binary_attention_mask = (attention_mask > -1.0) * 1.0
        token_score *= binary_attention_mask
        token_index = torch.argsort(token_score, dim=1, descending=True)
        token_rank = torch.argsort(token_index, dim=1)
        effective_token_length = torch.sum(binary_attention_mask, dim=1)
        if self.include_padding:
            effective_token_length = torch.ones_like(effective_token_length, device=effective_token_length.device) * token_score.size(1)
        token_rank = ((token_rank / (effective_token_length + 1e-6).unsqueeze(-1)).clamp(0.0, 1.0) * len(rank_mask)).detach().long().clamp_min(0)
        rank_mask_with_padding = torch.hstack([rank_mask, torch.tensor(0.0, device=rank_mask.device, dtype=rank_mask.dtype)])
        rank_keep_mask = rank_mask_with_padding[token_rank]
        # rank_keep_mask = torch.where(
        #     excluded_token_mask[..., 1:],
        #     torch.ones_like(rank_keep_mask),
        #     rank_keep_mask,
        # )
       
        # print(attention_mask.dtype)
        new_attention_mask = torch.where(
            rank_keep_mask == 0.0,
            -10000.0,
            attention_mask,
        )
        return new_attention_mask

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
        token_z=None,
        pruner_z=None,
        excluded_token_mask=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        B = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        if attention_mask is None:
            attention_mask = torch.ones([B,seq_len], device=device)
            
        # skip the first [CLS] token  and distill token
        prev_decision = torch.where(
            attention_mask[..., 1:].reshape((B, seq_len - 1, 1)) > -1,   
            torch.ones((B, seq_len - 1, 1), dtype=hidden_states.dtype, device=hidden_states.device),
            torch.zeros((B, seq_len - 1, 1), dtype=hidden_states.dtype, device=hidden_states.device),
        )
        
        last_decision = torch.where(
            attention_mask[..., 1:].reshape((B, seq_len - 1, 1)) > -1,   
            torch.ones((B, seq_len - 1, 1), dtype=hidden_states.dtype, device=hidden_states.device),
            torch.zeros((B, seq_len - 1, 1), dtype=hidden_states.dtype, device=hidden_states.device),
        )
        
        policy = torch.ones(
            B, seq_len, 1,
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        p_count = 0
        out_pred_prob = []
        pred_score = None
        if not self.training:
            constant_baseline_effective_lengths = torch.sum((attention_mask.reshape(B, -1) > -1.0), dim=1).detach().cpu().numpy()

        self.last_pred_score = None
        self.retain_mask = 1
        self.forward_mask = 0
        forward_hidden_states = hidden_states.clone()
        # print(attention_mask.dtype)
        # exit()
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.hard_token_mask is not None or (token_z is not None and len(token_z) != 0):
                assert len(self.token_prune_loc) > 0
                # enable token pruning
                if i in self.token_prune_loc:
                    if self.training and self.hard_token_mask is None:
                        # for training, apply soft mask on input tokens
                        hard_keep_decision = self.get_hard_keep_decision_for_training(
                            pred_score=pred_score,
                            rank_mask=token_z[p_count],
                            prev_decision=prev_decision,
                            attention_mask=attention_mask[..., 1:].reshape(B, -1),
                            pruner_mask=pruner_z[p_count],
                            # excluded_token_mask=excluded_token_mask,
                        )
                        out_pred_prob.append(hard_keep_decision.reshape(B, seq_len - 1))
                        cls_policy = torch.ones(B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                        # distill_token_policy = cls_policy = torch.ones(B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                        policy = torch.cat([cls_policy,hard_keep_decision], dim=1)
                        self.masks[i] = hard_keep_decision.squeeze(-1).detach() > 0
                        hidden_states *= policy       
                        layer_outputs = layer_module(
                            hidden_states,
                            attention_mask,
                            head_mask=layer_head_mask,
                            output_attentions=True,
                            intermediate_z=intermediate_z[i] if intermediate_z is not None else None,
                            head_z=head_z[i] if head_z is not None else None,
                            mlp_z=mlp_z[i] if mlp_z is not None else None,
                            head_layer_z=head_layer_z[i] if head_layer_z is not None else None,
                            hidden_z=hidden_z,
                        )
                        prev_decision = hard_keep_decision
                        if pruner_z[p_count] > 0:
                            last_decision  =  hard_keep_decision 
                            
                    else:
                        # for inference and post-finetuning, apply hard mask on attention_mask.
                        new_attention_mask = self.get_new_attention_mask_for_inference(
                            token_score=pred_score[:, :, 0],
                            rank_mask=token_z[p_count] if self.hard_token_mask is None else self.hard_token_mask[p_count],
                            attention_mask=attention_mask[..., 1:].reshape(B, -1),
                            pruner_mask=pruner_z[p_count] if self.hard_pruner_mask is None else self.hard_pruner_mask[p_count],
                            # excluded_token_mask=excluded_token_mask,
                        )
                        self.masks[i] = new_attention_mask.detach() > -1
                        attention_mask[..., 1:] = new_attention_mask.reshape(B, 1, 1, -1).clone()
                        layer_outputs = layer_module(
                            hidden_states,
                            attention_mask,
                            head_mask=layer_head_mask,
                            output_attentions=True,
                            intermediate_z=intermediate_z[i] if intermediate_z is not None else None,
                            head_z=head_z[i] if head_z is not None else None,
                            mlp_z=mlp_z[i] if mlp_z is not None else None,
                            head_layer_z=head_layer_z[i] if head_layer_z is not None else None,
                            hidden_z=hidden_z,
                        )
                        
                    p_count += 1
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        head_mask=layer_head_mask,
                        output_attentions=True,
                        intermediate_z=intermediate_z[i] if intermediate_z is not None else None,
                        head_z=head_z[i] if head_z is not None else None,
                        mlp_z=mlp_z[i] if mlp_z is not None else None,
                        head_layer_z=head_layer_z[i] if head_layer_z is not None else None,
                        hidden_z=hidden_z,
                    )
                # pred_score = layer_module.attention.self.token_score.unsqueeze(-1)
                # prev score
                # attention_probs = layer_outputs[1][1]
                # sz = attention_probs.shape[-1]
                # batch_size = attention_probs.shape[0]
                # # skip the first [CLS] token and [distill] token
                # pred_score = attention_probs.view(batch_size, -1, sz).mean(dim=1)[..., 1:].unsqueeze(-1)
                
                ################  new_score   ###############
                context = layer_outputs[1]
                attention_probs = layer_outputs[2]
                B,H,N,N = attention_probs.size()
                B,N,dim = context.size()
                h_dim = int(dim/H)
                context = context.view(B, H, N, h_dim)
                cls_attn = attention_probs[:, :, 0, 1:]  # [B, H, N-1]
                head_importance = context[..., 1:, :].norm(dim=-1)
                head_importance = head_importance / (head_importance.sum(dim=1, keepdim=True) + 1e-8)
                cls_attn = cls_attn * head_importance
                pred_score = cls_attn.sum(dim=1).unsqueeze(-1)  # [B, N-1,1]
                # pred_score = torch.sum(torch.abs(context), dim=-1)[..., 1:].unsqueeze(-1)
                
                
                self.pred_scores[i] = pred_score
                if not self.training:
                    self.inference_statistics["pruned_effective_lengths"][i] = torch.sum((attention_mask.reshape(B, -1) > -1.0), dim=1).detach().cpu().numpy()
                    self.inference_statistics["baseline_effective_lengths"][i] = constant_baseline_effective_lengths
            else:
                # disable token pruning
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask=layer_head_mask,
                    output_attentions=True,
                    intermediate_z=intermediate_z[i] if intermediate_z is not None else None,
                    head_z=head_z[i] if head_z is not None else None,
                    mlp_z=mlp_z[i] if mlp_z is not None else None,
                    head_layer_z=head_layer_z[i] if head_layer_z is not None else None,
                )
                # attention_probs = layer_outputs[1]
                # sz = attention_probs.shape[-1]
                # batch_size = attention_probs.shape[0]
                # # skip the first [CLS] token
                # pred_score = attention_probs.view(batch_size, -1, sz).mean(dim=1)[..., 1:].unsqueeze(-1)
                ################  new_score   ###############
                context = layer_outputs[1]
                attention_probs = layer_outputs[2]
                B,H,N,N = attention_probs.size()
                B,N,dim = context.size()
                h_dim = int(dim/H)
                context = context.view(B, H, N, h_dim)
                cls_attn = attention_probs[:, :, 0, 1:]  # [B, H, N-1]
                head_importance = context[..., 1:, :].norm(dim=-1)
                head_importance = head_importance / (head_importance.sum(dim=1, keepdim=True) + 1e-8)
                cls_attn = cls_attn * head_importance
                pred_score = cls_attn.sum(dim=1).unsqueeze(-1)  # [B, N-1,1]
                
                self.last_pred_score = pred_score
                self.pred_scores[i] = pred_score
            hidden_states = layer_outputs[0]
            

            if self.masks[i] is not None:
                mask_with_cls_distill = torch.ones(self.masks[i].shape[0], self.masks[i].shape[1]+1, device=self.masks[i].device)
                mask_with_cls_distill[:, 1:] = self.masks[i]
                self.retain_mask = mask_with_cls_distill.view(*mask_with_cls_distill.shape, 1)
                self.forward_mask = 1 - self.retain_mask
            forward_hidden_states = forward_hidden_states * self.forward_mask + hidden_states * self.retain_mask

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [forward_hidden_states, all_hidden_states, all_attentions] if v is not None)
        
        all_outputs = dict(
            encoder_outputs=BaseModelOutput(
                last_hidden_state=forward_hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
            ),
            token_pruning_outputs=None,
            prev_decision = last_decision
        )
        return all_outputs   


class PrunDeiTModel(DeiTModel):
    def __init__(self, config, token_prune_loc=None,use_mask_token: bool = False,add_pooling_layer: bool = True):
        super().__init__(config)
        self.encoder = PrunDeiTEncoder(config, token_prune_loc=token_prune_loc)
        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        attention_mask=None,
        token_z=None,
        pruner_z=None,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # if attention_mask is None:
        #     attention_mask = torch.ones(input_shape, device=device)       
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        device = pixel_values.device
       
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        
        B, token_lenth, _ = embedding_output.size()
        input_shape = torch.Size([B,token_lenth])
        # [B,token_length] -> [B,1,token_length,token_length]
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(   #in line 309 need to recover
            attention_mask, input_shape, device)
        
        extended_attention_mask.to(dtype=torch.float64) # fp32 -> fp 64 to compat get_inference_attention mask
        # print(extended_attention_mask.dtype)
        # exit()
        
        all_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            token_z=token_z,
            head_z=head_z,
            intermediate_z=intermediate_z,
            attention_mask = extended_attention_mask,
            pruner_z=pruner_z,
            return_dict = return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        encoder_outputs = all_outputs["encoder_outputs"]
        token_pruning_outputs = all_outputs["token_pruning_outputs"]
        prev_decision = all_outputs['prev_decision']

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # if not return_dict:
        #     head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
        #     return head_outputs + encoder_outputs[1:]
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        base_model_output_with_pooling = BaseModelOutputWithPooling_decision(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            prev_decision = prev_decision
        )
        return base_model_output_with_pooling, token_pruning_outputs
        # return BaseModelOutputWithPooling(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        # )
from transformers.file_utils import ModelOutput

class DeiTForImageClassificationWithTeacherOutputLoss(ModelOutput):
    logits: torch.FloatTensor = None
    cls_logits: torch.FloatTensor = None
    distillation_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss: torch.FloatTensor = None

class BaseModelOutputWithPooling_decision(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    prev_decision:torch.FloatTensor = None
    


class PrunDeiTModelForClassficationWithTeacher(DeiTPreTrainedModel):
    def __init__(self, config: DeiTConfig, token_prune_loc=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vit = PrunDeiTModel(config, token_prune_loc=token_prune_loc,add_pooling_layer=False)
        # Classifier heads
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        attention_mask=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_z=None,
        pruner_z=None,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs,token_pruning_outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            token_z=token_z,
            head_z=head_z,
            intermediate_z=intermediate_z,
            attention_mask = attention_mask,
            pruner_z=pruner_z,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prev_decision = outputs['prev_decision']

        logits = self.classifier(sequence_output[:, 0, :])
        feature_logits = self.classifier(sequence_output[:, 1:, :])

        loss = None
        # if labels is not None:
        #     #compute loss
        #     base_criterion = SoftTargetCrossEntropy()
        #     loss = base_criterion(logits,labels)
            
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss if loss is not None else None,
            logits=logits,
            feature_logits=feature_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            prev_decision = prev_decision
        )


# class PrunDeiTModelForClassfication(DeiTPreTrainedModel):
#     def __init__(self, config: DeiTConfig, token_prune_loc=None):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.deit = PrunDeiTModel(config, token_prune_loc=token_prune_loc,add_pooling_layer=False)
#         # self.deit = DeiTModel(config,add_pooling_layer=False)
#          # Classifier head
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

#         # Initialize weights and apply final processing
#         self.post_init()


#     def forward(
#         self,
#         pixel_values: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         token_z=None,
#         pruner_z=None,
#         head_z=None,
#         head_layer_z=None,
#         intermediate_z=None,
#         mlp_z=None,
#         hidden_z=None,
#     ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs,token_pruning_outputs= self.deit(
#             pixel_values,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             token_z=token_z,
#             pruner_z=pruner_z,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]
#         # print(sequence_output.shape)
#         logits = self.classifier(sequence_output[:, 0, :])
#         # we don't use the distillation token

#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return ImageClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

class ImageClassifierOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    feature_logits: torch.FloatTensor = None
    # cls_logits: torch.FloatTensor = None
    # distillation_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    prev_decision: torch.FloatTensor = None

def has_negative(tensor):
    return torch.lt(tensor, torch.zeros_like(tensor)).any()

# tensor = torch.tensor([1, 2, -3])
# print(has_negative(tensor))  # True
    