import pandas as pd
from collections import Counter
df1=pd.read_csv('C:/Users/hyeyoung/OneDrive - dongguk.edu/바탕 화면/10월_업무파일/data최종.csv',encoding='utf-8-sig')


def modefinder2(nums):
    c=Counter(nums)
    order=c.most_common()
    maximum=order[0][1]

    modes=[]
    for num in order:
        if num[1]==maximum:
            modes.append(num[0])
    return modes #  최빈값 2개이상 return해줌.


emotion_list = []
condition=(df1['Clip_no'])
for i in range(1, 10):
    condition = (df1['Clip_no']==i)
    #  특정 조건 만족하는 인덱스 구하기
    condition_index=df1[df1['Clip_no']==i].index
    for j in range(0, len(condition_index)):
        emotion_list.append(df1.iloc[j, 2])
    print(emotion_list)
    most_emotion=modefinder2(emotion_list)
    print(most_emotion)


