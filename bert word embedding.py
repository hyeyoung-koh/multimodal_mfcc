#bert:주변 단어에 의해 동적으로 변하는 단어 표현을 생성한다.
#Hugging Face로 bert용 pytorch 인터페이스를 설치한다.
#colab이라면
#!pip install transformers

#transformers는 BERT를 다른 작업(토큰 분류, 텍스트 분류 등)에 적용하기 위해 여러 클래스를 제공한다.
#이번 포스팅에서는 단어 임베딩이 목적이므로 출력이 없는 기본 BertModel을 사용한다.
import torch
from transformers import BertTokenizer,BertModel

import logging
#logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
% matplotlib inline

#load pre-trained model tokenizer(vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

#input formatting
#bert는 특정 형식의 입력 데이터를 필요로 한다.
#[sep]:문장의 끝을 표시하거나 두 문장을 분리할 때 사용
#[CLS]:문장 시작할 때.
#BERT에서 사용되는 단어사전에 있는 토큰
#BERT토크나이저의 토큰에 대한 Token ID
#시퀀스에서 어떤 요소가 토큰이고 패딩요소인지를 나타내는 Mask ID
#다른문장 구별하는 데 사용되는 SegmentID
#시퀀스 내에서 토큰 위치 표시하는 데 사용되는 Positional Embeddings
#transformers인터페이스는 위의 모든 사항을 처리:tokenizer.encode_plus함수 이용.
#[CLS]토큰은 항상 텍스트 시작 부분에 나타남&분류 문제 해결할 때만 사용 but다른 문제 풀더라도 입력은 무조건 해야함.
#tokenization
text="임베딩을 시도할 문장이다"
marked_text="[CLS]"+text+"[SEP]"
#tokenize our sentence
tokenized_text=tokenizer.tokenize(marked_text)
#print our tokens
print(tokenized_text)
##로 시작하는 토큰은 하위 단어 또는 개별 문자이다.
text = "배를 타고 여행을 간다." \
       "추석에 먹은 배가 맛있었다."

# Add the special tokens.
marked_text = "[CLS] " + text + " [SEP]"
#split sentence into tokens.
tokenized_text=tokenizer.tokenize(marked_text)
# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

for tup in zip(tokenized_text,indexed_tokens):
    print('{:<12} {:>6,}'.format(tup[0], tup[1]))
#SegmentID
#BERT는 두문장 구별 위해 1/0사용하여 문장 쌍을 학습하고 예상한다.
#우리의 목적 위해 단일 문장 입력에는 1리스트만 필요->입력 문장의 각 토큰에 대해 1로 구성된 벡터를 생성한다.
segments_ids=[1]*len(tokenized_text)
#running bert on our text
#데이터를 토치 센서로 변환하고 BERT모델을 호출해야한다.
#BERT Pytorch인터페이스에서는 데이터 형태가 python list가 아닌 토치 텐서가 필요하므로 이번 장에서 변환한다.
#convert inputs to pytorch tensors
tokens_tensor=torch.tensor([indexed_tokens])
segments_tensor=torch.tensor([segments_ids])
#model.eval()은 평가 모드로 모델을 설정한다.
#torch.no_grad는 PyTorch가 순방향 패스(forward pass)를 하는동안 컴퓨팅 그래프를 구성하지 않도록 한다(여기서는 backprop를 실행하지 않기 때문에).
with torch.no_grad():
    outputs=model(tokens_tensor,segments_tensors)
    hidden_states=outputs[2]
    #output_hidddn_states=True라 설정해서[2]는 hidden states from all layers이다.
print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
layer_i = 0 #13 layers

print ("Number of batches:", len(hidden_states[layer_i]))
batch_i = 0 #1 sentences

print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
token_i = 0 #36 tokens in our sentence

print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i])) #768 hidden units

#for the 5th token in our sentence,select its feature values from layer 5.
token_i=5
layer_i=5
vec=hidden_states[layer_i][batch_i][token_i]

plt.figure(figsize=(10,10))
plt.hist(vec,bins=200)
plt.show()

#레이어별로 값을 그룹화하는 것이 모델에 적합하지만, 단어 임베딩을 위해 토큰별로 그룹화한다.
#현재 차원:[#layers,#batches,#tokens,#features]
#원하는 차원:[#tokens,#layers,#features]
#pytorch에는 텐서 차원을 쉽게 재배열할 수 있는 permute함수가 포함되어있다.
#그러나 첫번째 차원은 현재 python list이다.
print('type of hidden_states:',type(hidden_states))
#each layer in the list is a torch tensor
print('Tensor shape for each layer:',hidden_states[0].size())
#concatenate the tensors for all layers.
#create a new dimension in tensor
token_embeddings=torch.stack(hidden_states,dim=0)
token_embeddings.size() # torch.Size([13,1,36,768])
#batches size는 필요하지 않으므로 제거한다.
token_embeddings=torch.squeeze(token_embeddings,dim=1) #remove dimension 1,the batches
token_embeddings.size() #torch.Size([13,36,768])
#마지막으로 permute사용해서 layers및 tokens차원을 전환할 수 있다.
token_embeddings=token_embeddings.permute(1,0,2)
token_embeddings.size() #torch.Size([36,13,768])
#각 토큰에 대한 개별 벡터 또는 전체 문장의 단일 벡터 표현을 얻고싶지만, 입력의 각 토큰에 대해 각각 768 크기의 13개의 개별 벡터가 있다.
#개별벡터 얻으려면 일부 레이어 벡터를 결합해야한다.








