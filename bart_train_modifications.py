import json
from datasets import load_metric,Dataset,DatasetDict,load_dataset
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
from torch.autograd import grad
import os
import torch
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES']="2"
os.environ['WANDB_DISABLED']="true"

model_checkpoint = "facebook/bart-large-cnn"
metric = load_metric("rouge.py")

TEST_SUMMARY_ID = 1


def transform_single_dialogsumm_file(file):
    data = open(file,"r").readlines()
    result = {"fname":[],"summary":[],"dialogue":[]}
    for i in data:
        d = json.loads(i)
        for j in d.keys():
            if j in result.keys():
                result[j].append(d[j])
    return Dataset.from_dict(result)

def transform_test_file(file):
    data = open(file,"r").readlines()
    result = {"fname":[],"summary%d"%TEST_SUMMARY_ID:[],"dialogue":[]}
    for i in data:
        d = json.loads(i)
        for j in d.keys():
            if j in result.keys():
                result[j].append(d[j])
    
    result["summary"] = result["summary%d"%TEST_SUMMARY_ID]
    return Dataset.from_dict(result)

def transform_dialogsumm_to_huggingface_dataset(train,validation,test):
    train = transform_single_dialogsumm_file(train)
    validation = transform_single_dialogsumm_file(validation)
    test = transform_test_file(test)
    return DatasetDict({"train":train,"validation":validation,"test":test})

import pandas as pd
df=pd.read_csv('ecir_train_data.csv')
val_df=pd.read_csv('ecir_val_data.csv')
val_df=val_df.drop_duplicates(subset="abstract",keep='first')
val_df.to_csv('val.csv')
print(len(df[df['avg']>400]))
print(val_df.shape)
df1=df.iloc[:2300,2:-1]
df2=df.iloc[2400:,2:-1]
df=df1.append(df2)
print(df.columns)
print(df.shape)
df.to_csv('ecir_train_data.csv')
raw_datasets=load_dataset("csv",data_files={"train":"ecir_train_data.csv"})

raw_datasets_val1=load_dataset("csv",data_files={"validation":"ecir_val_data.csv"})
raw_datasets_test=load_dataset("csv",data_files={"test":"ecir_test.csv"})
print(type(raw_datasets))

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_input_length = 
max_target_length = 135
# model.resize_token_embeddings(max_input_length)
def preprocess_function(examples):
    inputs = [doc for doc in examples["abstract"]]
    # print(inputs)
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,padding="max_length")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels1 = tokenizer(examples["summary1"], max_length=max_target_length, truncation=True,padding="max_length")
        labels2 = tokenizer(examples["summary2"], max_length=max_target_length, truncation=True,padding="max_length")
        labels3 = tokenizer(examples["summary3"], max_length=max_target_length, truncation=True,padding="max_length")
    model_inputs["labels1"] = labels1["input_ids"]
    model_inputs["labels2"] = labels2["input_ids"]
    model_inputs["labels3"] = labels3["input_ids"]
    return model_inputs

def preprocess_function_val(examples):
    inputs = [doc for doc in examples["abstract"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,padding="max_length")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels1 = tokenizer(examples["summary"], max_length=max_target_length, truncation=True,padding="max_length")
    model_inputs["labels"] = labels1["input_ids"]
    return model_inputs

def preprocess_function_test(examples):
    inputs = [doc for doc in examples["abstract"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,padding="max_length")

    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
tokenized_datasets_val1 = raw_datasets_val1.map(preprocess_function_val, batched=True)
tokenized_datasets_test = raw_datasets_test.map(preprocess_function_test, batched=True)
# print(tokenized_datasets_val5['validation'])

# print(raw_datasets['train'].column_names)
tokenized_datasets=tokenized_datasets.remove_columns(raw_datasets['train'].column_names)
tokenized_datasets_val1=tokenized_datasets_val1.remove_columns(raw_datasets_val1['validation'].column_names)
tokenized_datasets_test=tokenized_datasets_test.remove_columns(raw_datasets_test['test'].column_names)
# tokenized_datasets=tokenized_datasets.remove_columns(raw_datasets['train'].column_names)
print(tokenized_datasets)
print(tokenized_datasets_test)
batch_size = 14
label_names=['labels1','labels2','labels3']
args = Seq2SeqTrainingArguments(
    "BART-CNN1",
    evaluation_strategy = "epoch",
    learning_rate=7e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=50,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,
    save_strategy="epoch",
    metric_for_best_model="eval_rouge1",
    greater_is_better=True,
    seed=42,
    generation_max_length=max_target_length,
    label_names=label_names,
    eval_accumulation_steps=500,
)



data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

import nltk
import numpy as np
from torch import nn
from rouge_score import rouge_scorer
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}
global c
c=0    
import torch
from torch import nn
from transformers import Trainer

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

class ROUGELoss(nn.Module):
    def __init__(self):
        super(ROUGELoss, self).__init__()

    def forward(self, predictions, targets):
        # Calculate the ROUGE score between the predictions and targets
        # Implementation of the ROUGE score calculation is up to you
        with torch.no_grad():
            rouge_score = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            scores = rouge_score.score(predictions,targets)

        scores=torch.tensor(scores['rouge1'][2],requires_grad=True)

        # Return the negative ROUGE score as the loss
        return -scores
# model = torch.nn.DataParallel(model)
# loss_fct = nn.CrossEntropyLoss()

def calc_weights(values):
    total_magnitude = sum(abs(v) for v in values)
    weights = [abs(v) / total_magnitude for v in values]
    return weights

def weighted_average(values):
    weights = calc_weights(values)
    weighted_sum = sum(w * v for w, v in zip(weights, values))
    return weighted_sum


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args,**kwargs):
      super().__init__(*args,**kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # print("compute loss called")
        labels1 = inputs.pop("labels1")
        labels2 = inputs.pop("labels2")
        labels3 = inputs.pop("labels3")
        # forward pass
        # output1= model(inputs.get("input_ids"),inputs.get("attention_mask"),labels1)
        # output2= model(inputs.get("input_ids"),inputs.get("attention_mask"),labels2)
        # output3= model(inputs.get("input_ids"),inputs.get("attention_mask"),labels3)
        inputs['labels']=labels1
        output1= model(**inputs)
        inputs['labels']=labels2
        output2= model(**inputs)
        inputs['labels']=labels3
        output3=model(**inputs)
        # print(type(output1))
        logits1 = output1.get("logits")
        logits2 = output2.get("logits")
        logits3 = output3.get("logits")
        preds1 = nn.functional.softmax(logits1, dim=-1).argmax(dim=-1)
        y1 = tokenizer.batch_decode(sequences=preds1, skip_special_tokens=True)

        preds2 = nn.functional.softmax(logits2, dim=-1).argmax(dim=-1)
        y2 = tokenizer.batch_decode(sequences=preds2, skip_special_tokens=True)

        preds3 = nn.functional.softmax(logits3, dim=-1).argmax(dim=-1)
        y3 = tokenizer.batch_decode(sequences=preds3, skip_special_tokens=True)
        preds_real=model.generate(labels1)
        y_preds = tokenizer.batch_decode(sequences=preds_real, skip_special_tokens=True)
        # loss_fct = ROUGELoss()
        # loss=loss_fct.forward(y1[0],y_preds[0])
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        sum_sum=0
        for i in range(0,len(y_preds)):
            scores1 = scorer.score(y_preds[i],y1[i])
            scores2 = scorer.score(y_preds[i],y2[i])
            scores3 = scorer.score(y_preds[i],y3[i])    
            values=[]   
            values.append(scores1['rouge1'][2])
            values.append(scores2['rouge1'][2])
            values.append(scores3['rouge1'][2])
            result = weighted_average(values)
            sum_sum+=result
        sum_sum=sum_sum/len(y_preds)
        #Target ko generate Summary1 ka logits1, Summary 2 ka logits 2, summary 3 ka logits 3
        # compute custom loss (suppose one has 3 labels with different weights)
        #important
        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(logits1.view(-1,self.model.config.vocab_size),labels1.view(-1))
        loss2 = loss_fct(logits2.view(-1,self.model.config.vocab_size),labels2.view(-1))
        loss3 = loss_fct(logits3.view(-1,self.model.config.vocab_size),labels3.view(-1))
        loss_mid1=torch.add(loss1,loss2)
        loss=torch.add(loss3,loss_mid1)
        loss=torch.div(loss,3)
        #important
        rouge_loss=torch.tensor(sum_sum)
        rouge_loss=torch.mul(sum_sum,100)
        loss=torch.sub(loss,rouge_loss)

        return ((loss,) + outputs) if return_outputs else loss


# print("Model Device is :",model.get_device())
trainer = CustomTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets_val1["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()



list_val=[tokenized_datasets_test["test"]]#,tokenized_datasets_val2["test"],tokenized_datasets_val3["test"],tokenized_datasets_val4["test"],tokenized_datasets_val5["test"]]
final_decoded_preds=[]
for val_data in list_val:    
    out = trainer.predict(val_data,num_beams=4,max_length=128) #test
    # print(out.keys())
    predictions,_,_= out
    # print(metric) #validation metric

    # print(predictions)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after e ach sentence
    decoded_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    # decoded_labels = [" ".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    final_decoded_preds.append(decoded_preds)


# output summaries on test set
with open("Training_Rouge.txt","w") as f: 
    for j in final_decoded_preds:
        for i in j:
            # print(i)
            f.write(i.replace("\n","")+"\n")

list_val=[tokenized_datasets_val1["validation"]]#,tokenized_datasets_val2["test"],tokenized_datasets_val3["test"],tokenized_datasets_val4["test"],tokenized_datasets_val5["test"]]
final_decoded_preds=[]
for val_data in list_val:    
    out = trainer.predict(val_data,num_beams=6,max_length=128) #test
    # print(out.keys())
    predictions,_,_= out
    # print(metric) #validation metric

    # print(predictions)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after e ach sentence
    decoded_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    # decoded_labels = [" ".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    final_decoded_preds.append(decoded_preds)


# output summaries on test set
with open("Training_Rouge_VAL.txt","w") as f: 
    for j in final_decoded_preds:
        for i in j:
            # print(i)
            f.write(i.replace("\n","")+"\n")
