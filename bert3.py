!pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
!pip install pytorch-transformers
!pip install  torch==1.1.0
!pip install transformers==2.2.2
!pip install sentencepiece==0.1.82

import torch
import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

review_train, review_test, label_train, label_test = train_test_split(reviews, labels, random_state=100, test_size=0.3,
                                                                      shuffle=True)
review_train_sent = ['[CLS] ' + str(review) + ' [SEP]' for review in review_train]
review_test_sent = ['[CLS] ' + str(review) + ' [SEP]' for review in review_test]

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
tokenized_review_train = [tokenizer.tokenize(sent) for sent in review_train_sent]
tokenized_review_test = [tokenizer.tokenize(sent) for sent in review_test_sent]

train_review_token_id = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_review_train]
test_review_token_id = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_review_test]

max_len = 128
train_review_token_id = pad_sequences(train_review_token_id, maxlen=max_len, dtype='long', padding='post',
                                      truncating='post')
test_review_token_id = pad_sequences(test_review_token_id, maxlen=max_len, dtype='long', padding='post',
                                     truncating='post')

mask_train = []
attention_mask_train = []
for sent in train_review_token_id:
    mask_train = [float(i > 0) for i in sent]
    attention_mask_train.append(mask_train)

mask_test = []
attention_mask_test = []
for sent in test_review_token_id:
    mask_test = [float(i > 0) for i in sent]
    attention_mask_test.append(mask_test)

train_review_token_id = torch.tensor(train_review_token_id)
attention_mask_train = torch.tensor(attention_mask_train)
label_train = torch.tensor(label_train)

test_review_token_id = torch.tensor(test_review_token_id)
attention_mask_test = torch.tensor(attention_mask_test)
label_test = torch.tensor(label_test)

batch_size=32

#데이터들 묶어주고 데이터로더 형식으로 변환해주기
train_data=TensorDataset(train_review_token_id,attention_mask_train,label_train)
train_sampler=RandomSampler(train_data)
train_dataloader=DataLoader(train_data,batch_size=batch_size,sampler=train_sampler)

test_data=TensorDataset(test_review_token_id,attention_mask_test,label_test)
test_dataloader=DataLoader(test_data,batch_size=batch_size)

#GPU를 사용하기 위한 device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#모델 설정.
#모델설정
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.to(device)

#옵티마이저 설정
optimizer=AdamW(model.parameters(),
            lr=2e-5, #학습률
            eps=1e-8 #0으로 나누는 것을 방지하기 위한 epsilon값
 )
#에폭 수
epochs=3

#총 훈련 스텝배치 반복회수*에폭
total_steps=len(train_dataloader)*epochs

#정확도 계산 함수
def flat_accuracy(preds,labels):
    pred_flat=np.argmax(preds,axis=1).flatten()
    labels_flat=labels.flatten()
    return np.sum(pred_flat==labels_flat)/len(labels_flat)

#training 에폭만큼 반복
for epoch_i in range(0,epochs):
    print("")
    print("===Epoch{:}/{:}=====".format(epoch_i+1,epochs))
    print('Training...')

    #로스 초기화
    total_loss=0
    #훈련 모드로 변경
    model.train()
    #데이터로더에서 배치만큼 반복하여 가져옴.
    for step,batch in enumerate(train_dataloader):
        batch=tuple(t.to(device)for t in batch)
        b_input_ids,b_input_mask,b_labels=batch
        outputs=model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask,labels=b_labels)
        loss=outputs[0] #loss구한다.
        total_loss+=loss.item() # 총 로스 계산.
        loss.backward() #backward수행으로 gradient 계산
        optimizer.step() #gradient통해 가중치 파라미터 업데이트
        model.zero_grad() #그래디언트 초기화#we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes
    #평균 로스 계산
    avg_train_loss=total_loss/len(train_dataloader)
    print(avg_train_loss)
#이제 test
print("")
print("Running test...")
accuracies=[]
predictions=[]
#평가모드로 변경
model.eval()
#변수 초기화
eval_loss,eval_accuracy=0,0
nb_eval_steps,nb_eval_examples=0,0

#데이터로더에서 배치만큼 반복해서 가져온다.
for batch in test_dataloader:
    #배치를 gpu에 넣음.
    batch=tuple(t.to(device) for t in batch)
    #배치에서 데이터 추출
    b_input_ids,b_input_mask,b_labels=batch
    #그래디언트 계산 안함.
    with torch.no_grad():
        #forward
        outputs=model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask)
        logits=outputs[0]#로짓
        #cpu로 데이터 이동
        logits=logits.detach().cpu().numpy()
        label_ids=b_labels.to('cpu').numpy()
        true_labels.append(label_ids)
        predictions.append(logits)
        #출력 로짓과 라벨을 비교해서 정확도 계산
        tmp_eval_accuracy=flat_accuracy(logits,label_ids)
        eval_accuracy+=tmp_eval_accuracy
        nb_eval_steps+=1
        print('accuracy:{0:.5f}'.format(eval_accuracy/nb_eval_steps))
        accuracies.append(eval_accuracy/nb_eval_steps)
print('complete!')
#새로운 영화 테스트
def convert_input_data(sentences):
    tokenized_text = [tokenizer.tokenize(sent) for sent in sentences]
    Max_len = 128
    input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_text]
    input_ids = pad_sequences(input_ids, maxlen=Max_len, padding='post', truncating='post', dtype='long')
    attention_mask = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_mask.append(seq_mask)

    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_mask)

    return inputs, masks

def test_sentences(sentences):
    model.eval()
    inputs,masks=convert_input_data(sentences)
    b_input_ids=inputs.to(device)
    b_input_masks=masks.to(device)
    with torch.no_grad():
        outputs=model(b_input_ids,token_type_ids=None,attention_mask=b_input_masks)
        logits=outputs[0]
        logits=logits.detach().cpu().numpy()
        return logits
a=test_sentences(['재미있어요','재미없어요'])
a

np.argmax(a[0])
np.argmax(a[1])
