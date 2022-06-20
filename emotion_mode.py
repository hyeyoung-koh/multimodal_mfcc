import pandas as pd
from collections import Counter
df1=pd.read_csv('C:/Users/hyeyoung/OneDrive - dongguk.edu/바탕 화면/10월_업무파일/data최종.csv',encoding='utf-8-sig') #인코딩 ok

def modefinder2(nums): #nums: list이다.
    c=Counter(nums)
    order=c.most_common()
    maximum=order[0][1]
    modes=[]
    for num in order:
        if num[1]==maximum:
            modes.append(num[0])
    return modes #최빈값 2개이상 return

#condition=(df1['Clip_no'])
new_df=pd.DataFrame(index=range(1,6),columns=['most_emotion'])
#print(new_df)

most_emotion_list=[]

#for i in range(1, 5):
for i in range(1,5601): #나중에는 for i in range(1,len(df1)):
    if i==2005 or i==4016 or i==3038 or i==3101 or i==3109 or i==3166 or i==3173 or i==3174 or i==3177 or i==3193 or i==3251 \
            or i==3270 or i==3273 or i==3274 or i==3288 or i==3289 or i==3290 or i==3291 or i==3295 or i==3296 or \
            i==3299 or i==3301 or i==3305 or i==3312 or i==3315 or i==3316 or i==3320 or i==3331 or i==3333 or i==3337 or i==3356 \
            or i==3366 or i==3401 or i==3402 or i==3405 or i==3406 or i==3408 or i==3412 or i==3414 or i==3415 or i==3416 or i==3427 \
            or i==3429 or i==3435 or i==3437 or i==3440:
        continue
    condition = (df1['Clip_no']==i)
    condition_index=df1[df1['Clip_no']==i].index
    print(condition_index) #0,1,2,3,4,5,6,7,8,9
    emotion_list = []

    #print(len(condition_index)) #11
    for j in condition_index: #j=condition_index안의 인덱스 값

        emotion_list.append(df1['Multimodal_emotion'][j])

    print('emotion_list:', emotion_list)
    most_emotion = modefinder2(emotion_list)
    most_emotion_list.append(most_emotion)
    print('most_emotion:', most_emotion)
        #emotion_list.clear()
        #new_df.iloc[i] = most_emotion
        #print(new_df)
    #i+=1

print(most_emotion_list)

df_new=pd.read_csv('E:/emotion_mode_result.csv')
df_new.loc[:,'emotion_mode']=most_emotion_list
df_new.to_csv('E:/emotion_mode_final.csv')
