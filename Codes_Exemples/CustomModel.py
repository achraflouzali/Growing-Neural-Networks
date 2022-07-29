from metaeval import tasks_mapping, load_and_align
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import random
import os
from transformers import RobertaForSequenceClassification
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers.modeling_outputs import (SequenceClassifierOutput, TokenClassifierOutput)
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from src import unsuprisk
from typing import Dict
import math
import gc
import time
from src.flatness import QuotientManifoldTangentVector, riemannian_power_method, riemannian_hess_quadratic_form


from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    set_seed,
    speed_metrics,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)

from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    find_labels,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)



class CustomModel(RobertaForSequenceClassification):
    def __init__(self, config, model="distilroberta-base", growing = True):
        super(CustomModel,self).__init__(config)

        #Small Tweaks to adapt the models
        self.growing = growing

        if self.growing:
            self.classifier = GrowingClassificationHead(config, number_of_additions = 3)
        else:
            self.classifier = ClassificationHead(config)
        # Initialize weights and apply final processing
        self.roberta.init_weights()
        self.roberta._backward_compatibility_gradient_checkpointing()

    def growth(self):
        if self.growing:
            self.classifier.insertion()
            
    def get_weight_tensors(self):
        W=[]
        B=[]
        for i in range(len(self.roberta.encoder.layer)):
            W += [self.roberta.encoder.layer[i].attention.self.query.weight,
            self.roberta.encoder.layer[i].attention.self.key.weight,
            self.roberta.encoder.layer[i].attention.self.value.weight,
            self.roberta.encoder.layer[i].attention.output.dense.weight,
            self.roberta.encoder.layer[i].intermediate.dense.weight,
            self.roberta.encoder.layer[i].output.dense.weight]
            B += [self.roberta.encoder.layer[i].attention.self.query.bias,
            self.roberta.encoder.layer[i].attention.self.key.bias,
            self.roberta.encoder.layer[i].attention.self.value.bias,
            self.roberta.encoder.layer[i].attention.output.dense.bias,
            self.roberta.encoder.layer[i].intermediate.dense.bias,
            self.roberta.encoder.layer[i].output.dense.bias]
         """
                    Arborescence transformer style RoBERTa : 
                        - Couches de transformers (i) type "Roberta Layer"
                            - Attention
                                - self ("self-attention") : trainer.model.model.encoder.layer[i].attention.self
                                    - query (query.weight)
                                    - key (key.weight)
                                    - value (value.weight)
                                - output : trainer.model.model.encoder.layer[i].attention.output
                                    - dense (dense.weight)
                            - Intermediate : trainer.model.model.encoder.layer[i].intermediate
                                - dense (dense.weight)
                            - Output : trainer.model.model.encoder.layer[i].output
                                - dense (dense.weight)
                        -
            """
        W += self.classifier.get_weights()
        B += self.classifier.get_bias()
        weight = W+B
        return weight

  def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                  self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



