
#fine-tune BERT on the IMDB dataset->classify whether movie reviews positive/negative
from datasets import load_dataset

raw_datasets=load_dataset("imdb")
#train->training에 이용/ test->validation에 이용
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("bert-base-cased")
inputs=tokenizer(sentences,padding="max_length",truncation=True) #by padding,truncating=>512길이까지 accept가능함.

#전처리 과정
def tokenize_function(examples):
    return tokenizer(examples["text"],padding="max_length",truncation=True)
tokenized_datasets=raw_datasets.map(tokenize_function,batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]
#데이터 정의는 완료됨.

#model을 정의해보자.
from transformers import AutoModelForSequenceClassification
model=AutoModelForSequenceClassification.from_pretrained("bert-base-cased",num_labels=2)

#to define our Trainer,we need to instantiate a TrainingArguments.
#This class contains all hyperparameters we can tune for the Trainer or the flags to activate different training options it supports.
#only thing we have to provide is a directory in which checkpoints will be saved
from transformers import TrainingArguments
training_args=TrainingArguments("test_trainer")

from transformers import Trainer
#Trainer API: Trainer class에 학습에 필요한 하이퍼파라미터를 제공하면 자동으로 해당 설정에 맞게 학습을 진행할 수 있다.
#이때 hyperparameter는 TrainingArguments라는 클래스에 명시적으로 정의할 수 있다.
#이렇게 trainer를 정의했다면,trainer.train()을 통해서 자동으로 학습이 가능하다.
trainer=Trainer(model=model,args=training_args,train_dataset=small_train_dataset,
                eval_dataset=small_eval_dataset)
#To fine-tune our model, we just need to call trainer.train()
trainer.train()
#train이 될 것이다.
#there is no evaluation during training,and we didn't tell the Trainer to compute any metrics.
import numpy as np
from datasets import load_metric

metric=load_metric("accuracy")
def compute_metrics(eval_pred):
    logits,labels=eval_pred
    predictions=np.argmax(logits,axis=-1)
    return metric.compute(predictions=predictions,references=labels)
    #compute함수는 needs to receive a tuple and has to return a dictionary with string keys and float values

#let's create a new Trainer with our fine-tuned model
trainer=Trainer(
    model=model,args=training_args,train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,compute_metrics=compute_metrics,
)
trainer.evalute()
#it showed an accuracy of 87.5% in our case.
#if you want to fine-tune your model and regularly report evaluation metrics(for instance at the end of each epoch),here is how you should define your training arguments.
from transformers import TrainingArguments
training_args=TrainingArguments("test_trainer",evaluation_strategy="epoch")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#fine-tuning in native pytorch
#first, we need to define the dataloaders, which we will use to iterate over batches.
tokenized_datasets=tokenized_datasets.remove_columns(["text"])
tokenized_datasets=tokenized_datasets.rename_column("label","labels")
tokenized_datasets.set_format("torch")

small_train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset=tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

#dataloaders를 define해보자.
from torch.utils.data import DataLoader

train_dataloader=DataLoader(small_train_dataset,shuffle=True,batch_size=8)
eval_dataloader=DataLoader(small_eval_dataset,batch_size=8)

#next,we define our model
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

#optimizer와 learning rate scheduler가 빠졌다.
#default optimizer used by Trainer is AdamW이다.
from transformers import AdamW
optimizer=AdamW(model.parameters(),lr=5e-5)
#finally, learning rate scheduler used by default is just a linear decay from the maximum value to 0
from transformers import get_scheduler

num_epochs=3
num_training_steps=num_epochs*len(train_dataloader)
lr_scheduler=get_scheduler("linear",
                           optimizer=optimizer,num_warmup_steps=0,num_training_steps=num_training_steps)
import torch
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
#이제 훈련을 위한 준비는 다 되었다.
from tqdm.auto import tqdm
progress_bar=tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch={k:v.to(device) for k,v in batch.items()}
        outputs=model(**batch)
        loss=outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

#we accumulate the predictions at each batch before computing the final result when the loop is finished
metric=load_metric("accuracy")
model.eval()
#logits=logistic+probit 이다. 즉, log+odds이다.
for batch in eval_dataloader:
    batch={k:v.to(device) for k,v in batch.items()}
    with torch.no_grad(): #torch.no_grad는 gradient연산을 옵션을 끌 떄 사용하는 것이다.
        outputs=model(**batch)
    logits=outputs.logits #logits는 최종 sigmoid나 softmax를 지난기 전 단계, 즉, model의 raw output을 의미한다.
    predictions=torch.argmax(logits,dim=-1)
    metric.add_batch(predictions=predictions,references=batch["labels"])
metric.compute()
#-------------------------------------------------------------------------------------
#기초적인 bert는 transformers의 encoder부분을 활용한다.
# $pip install transformers
#버트 활용 준비-2가지 모듈을 불러와야한다.
#하나는 토크나이저이고, 다른 하나는 가중치를 가지고 있는 bert모델이다.
#주의점: 전처리의 경우 모델을 사전학습시킬 때와 동일한 규칙을 사용해야하므로 모델을 학습할 때 사용했던 것과 동일하게 사용해야한다.
#우선 토크나이저를 자동으로 다운로드하자.
from transformers import *
tokenizer=BertTokenizer.from_pretrained('bert-base-multilingual-cased')
#버트 문장 전처리
#문장을 버트에 활용하려면 활용하는 분야에 맞게 다양한 입력값으로 치환해야한다.
#여기서 버트는 일반적으로 3가지 입력값을 넣어줘야한다.
#1.input_ids:문장을 토크나이즈해서 인덱스 값으로 변환하는 것이다.일반적으로 버트에서는 단어를 subword단위로 변환시키는 word piece toknizer로 활용한다.
#2.attention mask:패딩된 부분에 대해 학습에 영향을 받지 않기 위해 처리해주는 입력값이다. 버트 토크나이저에서는 1은 어텐션에 영향을 받는 토큰을 나타내고,0은 영향을 받지 않는 토큰을 나타낸다.
#3.token_type_ids:두 개의 시퀀스 입력으로 활용할 때 0과 1로 문장의 토큰 값을 분리한다.[CLS]는 문장의 시작을 의미하고 [SEP]는 문장이 분리되는 부분을 의미하는 토큰이다.
#자주 사용되는 스페셜 토큰:[UNK],[MASK],[PAD],[SEP],[CLS]
#버트에 필요한 입력값의 형태 데이터를 변환시키는 코드를 직접 구현해도 되지만,huggingface의 Tokenizer라이브러리를 활용하면 좀 더 쉽고 빠르게 버트의 입력값을 구할 수 있다.
#encode_plus의 순서
#1.문장을 토크나이징한다.
#2.add_special_tokens를 True로 지정하면 토큰의 시작점에 [CLS]토큰,토큰의 마지막에 [SEP]토큰을 붙인다.
#3.각 토큰을 인덱스로 변환한다.
#4.max_length에 MAX_LEN최대 길이에 따라 문장의 길이를 맞추는 작업을 진행하고,pad_to_max_length기능을 통해 MAX_LEN의 길이에 미치지 못하는 문장에 패딩을 적용한다.
#5.return_attention_mask기능을 통해 어텐션 마스크를 생성한다.
#6.토큰 타입은 문장이 1개일 경우 0으로, 문장이 2개일 경우 0과 1로 구분해서 생성한다.
#6-1)문장이 바뀔때마다 0에서 1로 바뀐 후 다시 1에서 0으로 바뀐다.
#6-2)ex[0,0,0,0,1,1,1,1,1,1,0,0,0,0,0]:3문장이라는 것을 알 수 있다.
#encode_plus코드 구현:버트에 필요한 입력형태로 변환&문장을 최대길이에 맞게 패딩&결과값을 딕셔너리로 출력해줌.
def bert_tokenizer(sent,MAX_LEN):
    encoded_dict=tokenizer.encode_plut(
        text=sent,
        add_special_tokens=True,
        max_length=MAX_LEN,
        pad_to_max_length=True,
        return_attention_mask=True
    )
    input_ed=encoded_dict['input_ids']
    attention_mask=encoded_dict['attention_mask']
    token_type_id=encoded_dict['token_type_ids']
    return input_id,attention_mask,token_type_id










