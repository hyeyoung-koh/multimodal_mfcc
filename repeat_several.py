import os

#내 코드에 적용
def loop_directory(directory:str):
    for filename in os.listdir(directory): #  특정 디렉토리의 파일을 반복한다.
        if filename.endswith('.wav'):
            file_directory=os.path.join(directory, filename) #  상위디렉토리와 디렉토리 내의 파일을 결합한다.
            print(file_directory)


if __name__ == '__main__':
    i=0
    clip='clip_'+str(i)
    for i in range(1, 401):
        loop_directory('D:/0001-0400/0001-0400/clip_1/')
        i+=1