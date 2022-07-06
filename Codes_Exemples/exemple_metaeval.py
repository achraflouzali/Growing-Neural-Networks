from metaeval import tasks_mapping, load_and_align
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig


### tasks_mapping is a pandas DataFrame
### with columns : 'task_tuple' , 'text_fields', 'label_fields', 'split_keys',
###       'num_labels'
### We isolate the binary classification tasks (num_labels == 2) in binary_tasks_mapping
binary_tasks_mapping = tasks_mapping[tasks_mapping['num_labels']==2]


### Function to run experiments on a chosen_model (default : distilroberta-base) and a task (default : all the tasks).
### Beware : Bug with utilitarianism and amazon_polarity
def exp(chosen_model = "distilroberta-base", chosen_task = "total"):
    tokenizer = AutoTokenizer.from_pretrained(chosen_model)
    model = AutoModelForSequenceClassification.from_pretrained(chosen_model, num_labels=2)
    
    ### Function to tokenize the inputs, whether they have one or two sentences.
    def tokenize_function(examples):
        args = (
            (examples["sentence"],) if "sentence" in examples.keys() else (examples["sentence1"], examples["sentence2"])
        )
        return tokenizer(*args, padding="max_length", truncation=True)

    ### Whether to process all the binary classification tasks from metaeval or not :
    if chosen_task == "total":
        for i, task in binary_tasks_mapping['task_tuple']:
            ### We handle each dataset whether it has a default task or not
            if task in ['default', 'labeled_final', "plain_text"]:
                i, task = task, i
            
            ### Load an Tokenize the current task
            dataset = load_and_align(task)
            tokenized_datasets = dataset.map(tokenize_function, batched=True)

            ### Check if the task dataset is splitted in 3 (train, validation, test)
            if len(dataset.keys()) == 3:
                train, validation, test = tokenized_datasets['train'], tokenized_datasets['validation'], tokenized_datasets['test']
                
                ### Definition of the correct metric for each task : @TODO check whether it can be done better.
                metric = load_metric("accuracy")
                if chosen_task == "cola":
                    metric = load_metric("matthews_correlation")
                else:
                    metric = load_metric("accuracy")

                def compute_metrics(eval_pred):
                    logits, labels = eval_pred
                    predictions = np.argmax(logits, axis=-1)
                    return metric.compute(predictions=predictions, references=labels)        

                ### Definition of standard Training Arguments
                training_args = TrainingArguments("test_trainer",logging_strategy='epoch', evaluation_strategy="epoch", learning_rate=2e-05, num_train_epochs=15,save_strategy="epoch", eval_accumulation_steps=1, load_best_model_at_end=True, metric_for_best_model=metric.name) 
                
                trainer = Trainer(model=model, args=training_args, train_dataset=train, eval_dataset=validation,compute_metrics=compute_metrics)

                trainer.train()
                trainer.evaluate()
                
                ### Uncomment the following line for a generalization performances estimation on the test data-set.
                #trainer.predict(test)
                #print(test_results)
            else:
                pass

    ### If the user wants to work on a precise task. @TODO : Unify the case of a list of multiple tasks with script .sh           
    else:
        ### Test if the task exists in binary_tasks_mapping
        try:
            chosen_task in binary_tasks_mapping['task_tuple']
        except ValueError:
            print("The chosen task should be among the following : %s. By default, all tasks are considered."%binary_tasks_mapping['task_tuple'].index)
        
        ### Load an Tokenize the current task
        dataset = load_and_align(chosen_task)
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        
        ### Check if the task dataset is splitted in 3 (train, validation, test)
        if len(dataset.keys()) == 3:
            train, validation, test = tokenized_datasets['train'], tokenized_datasets['validation'], tokenized_datasets['test']
        
            ### Definition of the correct metric for the chosen task.
            metric = load_metric("accuracy")
            if chosen_task == "cola":
                metric = load_metric("matthews_correlation")
            else:
                metric = load_metric("accuracy")
        
            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                return metric.compute(predictions=predictions, references=labels)
        
            ### Definition of standard Training Arguments
            training_args = TrainingArguments("test_trainer",logging_strategy='epoch', evaluation_strategy="epoch", learning_rate=2e-05, num_train_epochs=15, save_strategy="epoch", eval_accumulation_steps=1, load_best_model_at_end=True, metric_for_best_model=metric.name)
        
            trainer = Trainer(model=model, 
                        args=training_args, train_dataset=train,
                        eval_dataset=validation,compute_metrics=compute_metrics)
            trainer.train()
            trainer.evaluate()
            
            ### Uncomment the following lines for a generalization performances estimation on the test data-set.
            #test_results = trainer.predict(test)
            #print(test_results)
        else:
            pass
