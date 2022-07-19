import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric
from metaeval import tasks_mapping, load_and_align
from transformers import AutoTokenizer
import wandb
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import datasets
from datasets import Dataset
import time

models=['bert-base-uncased','facebook/bart-base','roberta-large','distilroberta-base','t5-base']
total_tasks=['wic', 'cola', 'justice', 'utilitarianism', 'virtue', 'emobank-arousal',
 'persuasiveness-eloquence', 'persuasiveness-relevance', 'persuasiveness-specificity', 'persuasiveness-strength',
 'emobank-dominance', 'squinky-implicature', 'sarcasm', 'squinky-formality', 'squinky-informativeness',
 'emobank-valence', 'paws', 'imdb', 'deontology', 'sst2', 'mrpc', 'qqp', 'qnli', 'rte', 'wnli', 'recast_puns',
 'recast_factuality', 'recast_verbnet', 'recast_verbcorner', 'recast_ner', 'recast_sentiment',
 'recast_megaveridicality', 'boolq', 'commonsense', 'hate', 'irony', 'offensive', 'rotten_tomatoes',
 'subj_number', 'obj_number', 'past_present', 'coordination_inversion', 'odd_man_out',
 'bigram_shift', 'hover', 'movie_rationales', 'eraser_multi_rc',  'answer_selection_experiments']


v=time.strftime('%A %d/%m/%Y %H:%M:%S')
v=v.replace("/","-")
v=v.replace(":","-")
v='Monday 18-07-2022 14-45-34'
lr_l=[2e-05,2e-04,2e-03,2e-02,2e-01]
num_epochs=[5,10,15,20]
wandb.login()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def fusion_dataset(L):
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




def metaev(mission,lr,num_epochs,model_name,tasks):
        def compute_metrics(eval_pred):
           logits, labels = eval_pred
           predictions = np.argmax(logits, axis=-1)
           return metric.compute(predictions=predictions, references=labels)
        def tokenize_function(examples):
                args=0
                if "sentence1" in examples.keys():
                  if examples["sentence1"]=='':
                   args = (examples["sentence"],)
                  else:
                   args=(examples["sentence1"], examples["sentence2"])
                else:
                  args = (examples["sentence"],)
                return tokenizer(*args, padding="max_length", truncation=True)


        tokenizer = AutoTokenizer.from_pretrained(model_name)
        metric = load_metric("accuracy")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        if mission=='ub':
                for task in tasks:
                        run = wandb.init(project="metaeval   "+v,group='upper bound',name=task)
                        training_args = TrainingArguments(task,
                        logging_strategy='epoch',
                        evaluation_strategy="epoch",
                        learning_rate= lr,
                        num_train_epochs=num_epochs,
                        report_to='wandb'
                        )
                        dataset =load_and_align(task)
                        tokenized_dataset=dataset.map(tokenize_function,batched=True)
                        trainer = Trainer(model=model,
                                        args=training_args,
                                        train_dataset=tokenized_dataset["train"],
                                        eval_dataset=tokenized_dataset["validation"],
                                        compute_metrics=compute_metrics
			)
                        trainer.train()
                        trainer.evaluate()
                        run.finish()
        if mission=='lb':
                run=wandb.init(project='metaeval   '+v,group='lower bound',name='all_tasks')
                training_args = TrainingArguments('all tasks',
                                    logging_strategy='epoch',
                                    evaluation_strategy="epoch",
                                    learning_rate= lr,
                                    num_train_epochs=num_epochs,
                                    report_to="wandb")
                dataset=fusion_dataset(tasks)
                tokenized_dataset=dataset.map(tokenize_function,batched=True)
                trainer = Trainer(      model=model,
                                        args=training_args,
                                        train_dataset=tokenized_dataset["train"],
                                        eval_dataset=tokenized_dataset["validation"],
                                        compute_metrics=compute_metrics
                                        )
                trainer.train()
                trainer.evaluate()
                run.finish()


def runexp(tasks):
     metaev('ub',2e-05,5,'bert-base-uncased',tasks)
     metaev('lb',2e-05,5,'bert-base-uncased',tasks)




runexp(['wic','cola','commonsense'])
