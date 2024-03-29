import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric
from metaeval import tasks_mapping, load_and_align
import wandb
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AdamW, AutoTokenizer,get_cosine_schedule_with_warmup
import datasets
from datasets import Dataset
import time
from transformers import Trainer
import torch
from transformers.utils import logging
import math
import numpy as np
from typing import Dict
from transformers.optimization import Adafactor, AdafactorSchedule



models=['bert-base-uncased','facebook/bart-base','roberta-large','distilroberta-base','t5-base']
total_tasks=['wic', 'movie_rationales', 'justice', 'utilitarianism', 'virtue', 'emobank-arousal',
 'persuasiveness-eloquence', 'persuasiveness-relevance', 'persuasiveness-specificity', 'persuasiveness-strength',
 'emobank-dominance', 'squinky-implicature', 'sarcasm', 'squinky-formality', 'squinky-informativeness',
 'emobank-valence', 'paws', 'imdb', 'deontology', 'sst2', 'mrpc', 'qqp', 'qnli', 'rte', 'wnli', 'recast_puns',
 'recast_factuality', 'recast_verbnet', 'recast_verbcorner', 'recast_ner', 'recast_sentiment',
 'recast_megaveridicality', 'boolq', 'commonsense', 'hate', 'irony', 'offensive', 'rotten_tomatoes',
 'subj_number', 'obj_number', 'past_present', 'coordination_inversion', 'odd_man_out',
 'bigram_shift', 'hover', 'cola', 'eraser_multi_rc',  'answer_selection_experiments']   # binary tasks with train,validation and test sets

v=time.strftime('%d-%m--%-H-%M')
lr_l=[2e-05,2e-04,2e-03,2e-02,2e-01]
num_epochs=[5,10,15,20]
wandb.login()                                                                          # you have to create a W&B account to log the learning data



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def fusion_dataset(L):
    ''' Function to merge and shuffle datasets of binary tasks '''
  l_train=[]
  l_validation=[]
  l_test=[]
  df=[]
  i=1
  if L=='total':
    tasks = total_tasks
  else:
    tasks=L.copy()
  j=len(tasks)
  for task in tasks:
    print(f"TASK {i}/{j}")
    dataset=load_and_align(task)
    l_train.append(pd.DataFrame(dataset['train']))
    l_validation.append(pd.DataFrame(dataset['validation']))
    l_test.append(pd.DataFrame(dataset['test']))
    i+=1
  df_train=pd.concat(l_train).sample(frac = 1).reset_index(drop=True).fillna('')
  df_validation=pd.concat(l_validation).sample(frac = 1).reset_index(drop=True).fillna('')
  df_test=pd.concat(l_test).sample(frac = 1).reset_index(drop=True).fillna('')
  if "sentence" in df_train.keys() and ('sentence1' in df_train.keys()):
    df_train=df_train[['sentence','sentence1','sentence2','label']]
    df_validation=df_validation[['sentence','sentence1','sentence2','label']]
    df_test=df_test[['sentence','sentence1','sentence2','label']]
  return datasets.DatasetDict({"train":Dataset.from_pandas(df_train),
                               "validation":Dataset.from_pandas(df_validation),
                               "test":Dataset.from_pandas(df_test)})



logger = logging.get_logger(__name__)

class CustomTrainer(Trainer):
    def __init__(self, input_size, task, prior0 = 0.5, device="cuda", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p0 = prior0
        self.task = task
        self.device = device
        self.scores = []
        self.risk = 0
        self.risk1 = 0
        self.input_size = input_size

    def binrisk(self, mu0, mu1, var0, var1, prior0, device):
        with torch.set_grad_enabled(True):
            sq2 = torch.tensor(math.sqrt(2.)).to(device)
            sigma0 = torch.sqrt(var0)
            sigma1 = torch.sqrt(var1)
            nor0 = torch.distributions.normal.Normal(mu0,sigma0)
            mor0 = torch.exp(nor0.log_prob(torch.tensor([-1.]).to(device)))
            nor1 = torch.distributions.normal.Normal(mu1,sigma1)
            mor1 = torch.exp(nor1.log_prob(torch.tensor([1.]).to(device)))
            prior1 = 1.-prior0

            m = mu0+1.
            r = torch.mul(prior0/2.,m)
            mm = -mu0-1.
            nn = torch.mul(sq2,sigma0)
            mm = torch.div(mm,nn)
            mm = torch.erf(mm)
            mm = 1.-mm
            term1 = torch.mul(r,mm)
            r = term1

            term2 = torch.mul(prior0,var0)
            term2 = torch.mul(term2,mor0)
            r = r+term2

            m3 = 1.-mu1
            term3 = torch.mul(prior1/2.,m3)
            nn3 = torch.mul(sq2,sigma1)
            mm3 = torch.div(m3,nn3)
            mm3 = 1. + torch.erf(mm3)
            term3 = torch.mul(term3,mm3)
            r = r+term3

            term4 = torch.mul(prior1,var1)
            term4 = torch.mul(term4,mor1)
            r = r+term4
            return r

    def preparePass(self):
        self.scores = np.sort(self.scores)
        n = int(self.p0 * float(len(self.scores)))
        self.n0 = float(n)
        self.n1 = float(len(self.scores)-n)
        # normalize the scores so that p0 are negative and (1-p0) positive
        self.lossbias = self.scores[n]
        for i in range(len(self.scores)):
            self.scores[i] -= self.lossbias
        self.mu0 = np.sum(self.scores[0:n])
        self.mu0 /= float(n)
        self.mu1 = np.sum(self.scores[n:])
        self.mu1 /= float(len(self.scores)-n)
        self.var0 = np.sum([x*x for x in self.scores[0:n]]) / float(n) - self.mu0*self.mu0
        self.var1 = np.sum([x*x for x in self.scores[n:]]) / float(len(self.scores)-n) - self.mu1*self.mu1
        if True:
            self.L = self.Lexact
        

    def Lexact(self,x):
        xx = x - self.lossbias
        xx = xx.view(-1)
        tv0 = torch.tensor(self.var0).to(self.device)
        tv1 = torch.tensor(self.var1).to(self.device)
        tp0 = torch.tensor(self.p0).to(self.device)
        tp1 = torch.tensor(1-self.p0).to(self.device)
        r = 0.
        r1 = 0.
        for i in range(xx.size(0)):
            if xx[i]<0:
                mm0 = self.mu0*self.n0
                tmu0 = xx[i] + mm0
                tmu0 = tmu0 / (self.n0+1.)
                tmu1 = torch.tensor(self.mu1).to(self.device)
            else:
                mm1 = self.mu1*self.n1
                tmu1 = xx[i] + mm1
                tmu1 = tmu1 / (self.n1+1.)
                tmu0 = torch.tensor(self.mu0).to(self.device)
            r += self.binrisk(tmu0,tmu1,tv0,tv1,tp0,self.device)
            r1 += self.binrisk(tmu0,tmu1,tv0,tv1,tp1,self.device)
        r /= float(xx.size(0))
        r1 /= float(xx.size(0))
        return r, r1

    def compute_unsup_risk(self, inputs):
        if len(self.scores) == self.input_size:
            #compute Unsup Risk after all the inputs have been processed
            self.preparePass()
            #Compute both the Unsup Risk with p0 and (1-p0)
            self.risk,self.risk1 = self.Lexact(inputs['input_ids'])[0].item(),self.Lexact(inputs['input_ids'])[1].item()
            #Reset the scores for the next epoch.
            self.scores = []
        else:
            #Nothing to do here.
            pass
    def compute_loss(self, model, inputs, return_outputs=False, unsup = True):
            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.
            Subclass and override for custom behavior.
            """
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs = model(**inputs)
            if unsup:
                if outputs.logits.size(-1)==1:
                    if self.device=="cuda": self.scores += outputs.logits.view(-1).cpu().detach().numpy()
                    else: self.scores += outputs.logits.view(-1).detach().numpy()
                else:
                    if outputs.logits.size(-1)==2:
                        if self.device=="cuda": self.scores += [z[0] for z in outputs.logits.view(-1,2).cpu().detach().numpy()]
                        else: self.scores += [z[0] for z in outputs.logits.view(-1,2).detach().numpy()]
                    else:
                        print("ERROR: must be binary !",outputs.logits.size())
                self.compute_unsup_risk(inputs)

            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            return (loss, outputs) if return_outputs else loss



    def log(self, logs: Dict[str, float]) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch

        logs["eval_risk-p0"] = self.risk
        logs["risk-(1-p0)"] = self.risk1
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)



        
        
def metaev(mission,lr,num_epochs,model_name,tasks):
    '''Function to finetune a model on tasks : upper bound to fine tune the model to the tasks independently, lower bound to fine tune the model to the dataset of 
        of every task combined'''
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        def compute_metrics(eval_pred):
           logits, labels = eval_pred
           predictions = np.argmax(logits, axis=-1)
           return metric.compute(predictions=predictions, references=labels)

        def tokenize_function(examples):
           args = ((examples["sentence"],) if "sentence" in examples.keys() else (examples["sentence1"], examples["sentence2"]))
           return tokenizer(*args, padding="max_length", truncation=True)


        metric = load_metric("accuracy")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        if mission=='ub':
                for task in tasks:
                        run = wandb.init(project="model= "+ model_name+" lr= "+str(lr)+"  nepoch= "+str(num_epochs),group='   '+v,name=task)   #to create a unique run each time in W&B with specifying the model,learning rate, the number of epoch and the date and time 
                        training_args = TrainingArguments(task,
                        evaluation_strategy="epoch",
                        save_strategy="no",
                        learning_rate= lr,
                        num_train_epochs=num_epochs,
                        weight_decay=0.001,
                        auto_find_batch_size=True,
                        report_to='wandb'
                        )
                        dataset =load_and_align(task)
                        tokenized_dataset=dataset.map(tokenize_function,batched=True)
                        train= tokenized_dataset["train"]
                        eval = tokenized_dataset["validation"]
                        trainer = CustomTrainer(model=model,input_size=len(train),task = task,prior0=train['label'].count(0)/len(train), args=training_args, train_dataset=train, eval_dataset=eval,compute_metrics=compute_metrics)
                        trainer.train()
                        trainer.evaluate()
                        run.finish()
        if mission=='lb':
                run = wandb.init(project="model= "+ model_name+" lr= "+str(lr)+"  nepoch= "+str(num_epochs),group='   '+v,name='all_tasks')
                training_args = TrainingArguments(task,                            
                        evaluation_strategy="epoch",
                        save_strategy="no",
                        learning_rate= lr,
                        num_train_epochs=num_epochs,
                        weight_decay=0.001,
                        auto_find_batch_size=True,
                        report_to='wandb'
                        )
                dataset=fusion_dataset(tasks)
                tokenized_dataset=dataset.map(tokenize_function,batched=True)
                trainer = Trainer(      model=model,
                                        args=training_args,
                                        train_dataset=tokenized_dataset["train"],
                                        eval_dataset=tokenized_dataset["validation"],
                                        compute_metrics=compute_metrics)
                trainer.train()
                trainer.evaluate()
                run.finish()


metaev('ub',3e-05,15,'bert-base-uncased',['sst2','mrpc','qqp','qnli', 'rte', 'wnli','wic','boolq','virtue','hover'])

