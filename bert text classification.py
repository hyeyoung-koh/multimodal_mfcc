import matplotlib.pyplot as plt
import pandas as pd
import torch

from torchtext.data import Field,TabularDataset,BucketIterator,Iterator

import torch.nn as nn
from transformers import BertTokenizer,BertForSequenceClassificaction

#training
import torch.optim as optim

#evaluation
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import seaborn as sns
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_SEQ_LEN=128
PAD_INDEX=tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX=tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

#Fields
#텍스트필드->뉴스 기사를 포함하는 데 사용됨.
label_field=Field(sequential=False,use_vocab=False,batch_first=True,dtype=torch.float)
text_field=Field(use_vocab=False,tokenizer=tokenizer.encode,lower=False,include_length=False,
                 batch_first=True,fix_length=MAX_SEQ_LENGTH,pad_token=PAD_INDEX,unk_token=UNK_INDEX)
fields=[('label',label_field),('title',text_field),('text',text_field),('titletext',text_field)]
#TabularDataset
train,valid,test=TabularDataset.splits(path=source_folder,train='train.csv',validationd='valid.csv',
                                       test='test.csv',format='CSV',fields=fields,skip_header=True)
#iterators
#Torchtext는 BucketIterator를 제공한다. 모든 텍스트 작업을 일괄로 처리하고 단어를 인덱스 숫자로 변환하는 것을 돕는다.
#이때,batch_size,device(CPU or GPU),shuffle과 같은 유용한 인자를 제공한다.
train_iter=BucketIterator(train,batch_size=16,sort_key=lambda x:len(x.text),
                          device=device,train=True,sort=True,sort_within_batch=True)
valid_iter=BucketIterator(valid,batch_size=16,sort_key=lambda x:len(x.text),
                          device=device,train=True,sort=True,sort_within_batch=True)
test_iter=Iterator(test,batch_size=16,device=device,train=False,shuffle=False,sort=False)

#모델 구축
class BERT(nn.Module):
    def __init__(self):
        super(BERT,self).__init__()
        options_name="bert-base-uncased"
        self.encoder=BertForSequenceClassification.from_pretrained(options_name)
    def forward(self,text,label):
        loss,text_fea=self.encoder(text,labels=label)[:2]
        return loss,text_fea

#훈련
#save and load functions
def save_checkpoint(save_path,model,valid_loss):
    if save_path==None:
        return
    state_dict={'model_state_dict':model.state_dict(),
                'valid_loss':valid_loss}
    torch.save(state_dict,save_path)
    print(f'Model saved to-->{save_path}')

def load_checkpoint(load_path,model):
    if load_path==None:
        return
    state_dict=torch.load(load_path,map_location=device)
    print(f'Model loaded from<=={load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

def save_metrics(save_path,train_loss_list,valid_loss_list,global_steps_list):
    if save_path==None:
        return
    state_dict={'train_loss_list':train_loss_list,
                'valid_loss_list':valid_loss_list,
                'global_steps_list':global_steps_list}
    torch.save(state_dict,save_path)
    print(f'Model saved to ==>{save_path}')

def load_metrics(load_path):
    if load_path==None:
        return
    state_dict=torch.load(load_path,map_location=device)
    print(f'Model loaded from<=={load_path}')
    return state_dict['train_loss_list'],state_dict['valid_loss_list'],state_dict['global_steps_list']

#training function
def train(model,optimizer,criterion=nn.BCELoss(),train_loader=train_iter,valid_loader=valid_iter,num_epochs=5,
          eval_every=len(train_iter)//2,file_path=destination_folder,best_valid_loss=float("Inf")):#가짜뉴스 탐지는 2가지 문제이므로 BinaryCrossEntropy를 손실함수로 사용한다.
    #initialize running values
    running_loss=0.0
    valid_running_loss=0.0
    global_step=0
    train_loss_list=[]
    valid_loss_list=[]
    global_steps_list=[]

    #training loop
    model.train()
    for epoch in range(num_epochs):
        for (labels,title,text,titletext),_ in train_loader:
            labels=labels.type(torch.LongTensor)
            labels=labels.to(device)
            titletext=titletext.type(torch.LongTensor)
            titletext=titletext.to(device)
            output=model(titletext,labels)
            loss,_=output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #update running values
            running_loss+=loss.item()
            global_step+=1

            #evaluation step
            if global_step%eval_every==0:
                model.eval()
                with torch.no_grad():
                    #validation loop
                    for (labels,title,text,titletext),_ in valid_loader:
                        labels=labels.type(torch.LongTensor)
                        labels=labels.to(device)
                        titletext=titletext.type(torch.LongTensor)
                        titletext=titletext.to(device)
                        output=model(titletext,labels)
                        loss,_=output
                        valid_running_loss+=loss.item()
                #evaluation
                average_train_loss=running_loss/eval_every
                average_valid_loss=valid_running_loss/len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                #resetting running values
                running_loss=0.0
                valid_running_loss=0.0
                model.train()

                #print progress
                print('Epoch [{}/{}],Step [{}/{}],Train Loss:{:.4f},Valid Loss:{:.4f}'.format(epoch+1,num_epochs,global_step,num_epochs*len(train_loader),
                                                                                              average_train_loss,average_valid_loss))
                #checkpoint
                if best_valid_loss>average_valid_loss:
                    best_valid_loss=average_valid_loss
                    save_checkpoint(file_path+'/'+'model.pt',model,best_valid_loss)
                    save_metrics(file_path+'/'+'metrics.pt',train_loss_list,valid_loss_list,global_steps_list)

        save_metrics(file_path+'/'+'metrics.pt',train_loss_list,valid_loss_list,global_steps_list)
        print('Finished Training!')

    model=BERT().to(device)
    optimizer=optim.Adam(model.parameters(),lr=2e-5)

    train(model=model,optimizer=optimizer)

