import torch
from torch import nn
from transformers import Trainer
import json
from datasets import load_metric,Dataset,DatasetDict,load_dataset
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
from torch.autograd import grad
import os
import torch
import pandas as pd


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
