from datasets import load_dataset
#import load_dataset
#import seaborn as sns
#import pandas as pd
#raw_datasets=pd.read_csv("imdb.csv")
raw_datasets=load_dataset("imdb")

from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("bert-base-cased")

inputs=tokenizer(sentences,padding="max_length",truncation=True)

def tokenize_function(examples):
    return tokenizer(examples["text"],padding="max_length",truncation=True)
tokenized_datasets=raw_datasets.map(tokenize_function,batched=True)

tokenized_datasets=tokenized_datasets.remove_columns(["text"])
tokenized_datasets=tokenized_datasets.rename_column("label","labels")
tokenized_datasets.set_format("torch") #데이터를 torch Tensor 형태로 바꿔준다.

small_train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset=tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

from torch.utils.data import DataLoader
#DataLoader:데이터 순서를 섞어서 8개씩 데이터를 반환하는 DataLoader이다.
#DataLoader의 필수 입력 인자는 DataSet이다.
#DataLoader객체는 학습에 쓰일 데이터 전체를 보관했다가 train함수가 batch하나를 요구하면 batch size개수만큼 데이터를 꺼내서 준다고 보면 된다.
#train()함수가 데이터를 요구하면 사전에 저장된 batch size만큼 return하는 형태이다.
#dataset으로 유의미한 data를 뽑아오기 위한 것이 DataLoader이다.
train_dataloader=DataLoader(small_train_dataset,shuffle=True,batch_size=8)
#여기서 batch size는 통상적으로 2의 제곱수로 설정한다. 각 minibatch의 크기를 말한다.(한 번의 배치 안에 있는 샘플 사이즈)
#shuffle=True:epoch마다 데이터셋을 섞어 데이터가 학습되는 순서를 바꾼다.
eval_dataloader=DataLoader(small_eval_dataset,batch_size=8)

#define our model
from transformers import AutoModelForSequenceClassification
model=AutoModelForSequenceClassification.from_pretrained("bert-base-cased",num_labels=2)

#이제 optimizer와 learning rate scheduler를 정의하자.
from transformers import AdamW
#학습하려는 모델의 매개변수와 학습률(learning rate)하이퍼파라미터를 등록하여 옵티마이저를 초기화한다.
optimizer=AdamW(model.parameters(),lr=5e-5)

from transformers import get_scheduler

num_epochs=3
num_training_steps=num_epochs*len(train_dataloader)
#learning rate scheduler used by default is just a linear decay from maximum value(5e-5) to 0
lr_scheduler=get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps)
import torch

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
#train준비가 끝남.

from tqdm.auto import tqdm
progress_bar=tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch={k:v.to(device) for k,v in batch.items()} #train_features,train_labels가 들어있다.
        outputs=model(**batch)
        #model(**batch)를 통해서 loss를 구하고 gradient를 계산한다.
        loss=outputs.loss
        loss.backward() #compute gradients of all variables wrt loss

        optimizer.step() #perform updates using calculated gradients
        lr_scheduler.step()
        optimizer.zero_grad() #clear previous gradients
#pytorch에서는 gradients값들을 추후에 backward를 해줄 때 계속 더해준다. 따라서,항상 backpropagation을 하기 전에 gradients를 0으로 만들어주고 시작해야한다.
#loss.backward()를 호출할 때 초기설정은 매번 gradient를 더해주는 것으로 설정되어있다.
        progress_bar.update(1)

metric=load_metric("accuracy")

model.eval()
#.eval()함수는 evaluation과정에서 사용하지 않아야하는 layer들을 알아서 off시키도록 하는 함수이다.
#evaluation/validation과정에선 보통 model.eval(),torch.no_grad()를 함께 사용한다고 한다.
for batch in eval_dataloader:
    batch={k:v.to(device) for k,v in batch.items()}
    with torch.no_grad():
    #torch.no_grad()는 gradient연산을 옵션을 끌 때 사용하는 파이썬 컨텍스트 매니저이다.
        outputs=model(**batch)
    logits=outputs.logits
    predictions=torch.argmax(logits,dim=-1)
    metric.add_batch(predictions=predictions,references=batch["labels"])
    #datasets.Metric.add_batch() are used to add pairs of predictions/reference to a temporary and memory efficient cache table.
metric.compute()

