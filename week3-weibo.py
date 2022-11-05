import jieba
import re
import os
import time
from tqdm import tqdm
import numpy as np
from scipy.spatial import distance

def ReadData(file_path):
    '''
    数据读取函数
    传入txt文件绝对路径
    返回列表(列表每个元素为time、location、text、emotion构成的字典)
    '''
    f=open(file_path,'r',encoding='utf8')
    data=[]
    f.readline()#去表头
    for line in tqdm(f.readlines()):
        text=''
        t=''
        location=[]
        #处理text
        message=line.split()
        if message[-2][0]!='+': continue#排除不含时间的错误数据
        for each in message[2:-7]:
            text+=' '+each
        #处理location
        if IsFloat(message[0][1:-2])==0:
            continue
        location.append(float(message[0][1:-2]))
        location.append(float(message[1][:-2]))
        #处理time
        for each in message[-6:]:
            t+=' '+each
        tt=time.strptime(t,' %a %b %d %H:%M:%S %z %Y')
        data.append({'time':time.mktime(tt),'location':location,'text':text,'emotion':'none'})
    f.close()
    return data

def IsFloat(number):
    '''
    判断是否是浮点数
    '''
    for each in number: 
        if each not in {'1','2','3','4','5','6','7','8','9','0','.'}: 
            return 0
    return 1

def CleanData(data):
    '''
    清洗数据, 去除重复值和噪声(url
    传入列表data(列表每个元素为time、location、text、emotion构成的字典)
    返回清洗后的列表(text的值为词构成的列表)
    '''
    res=[]
    url=re.compile(r'[http|https]*://[a-zA-Z0-9.?/&=:]*')#取url
    for each in tqdm(data):
        each['text']=re.sub(url,'',each['text'])#去url
        if each not in res:#去重
            res.append(each)
    return res

def CutText(data,dict_path):
    '''
    将data中的text分词
    传入列表data(列表每个元素为time、location、text、emotion构成的字典)和词典地址
    返回text分词后的列表
    '''
    f=open(dict_path,'r',encoding='utf8')
    stop_words = [line.strip() for line in f.readlines()]
    jieba.load_userdict(dict_path)#载入词典
    for each in tqdm(data):
        words=[]
        for word in list(jieba.cut(each['text'])):
            if word not in stop_words:#过滤停用词
                words.append(word)
        each['text']=words
    f.close()
    return data

def LoadEmotion(root):
    '''
    外函数加载情绪词
    传入情绪词文件的根目录
    返回情绪分析函数
    '''
    #载入词典
    emotions={}
    for root,dirs,files in os.walk(root):
        for file in files:
            emotion_name=file[:-4]
            emotion_words={line.strip() for line in open(root+'\\'+file,'r',encoding='utf8')}
            emotions.update({emotion_name:emotion_words})
    def Emotion(data):
        '''
        内函数统计情绪(引用MaxEmotion函数)
        传入列表data(列表每个元素为time、location、text、emotion构成的字典)
        返回修改过emotion的data列表
        '''
        for line in tqdm(data):
            nonlocal emotions#将载入的情绪词持久化
            count_emotion={each:0 for each in emotions}
            for word in line['text']:
                for emotion in emotions:
                    if word in emotions[emotion]: count_emotion[emotion]+=1
            line['emotion']=MaxEmotion(count_emotion)
        return data
    return Emotion

def MaxEmotion(dict):
    '''
    求字典最大值的键
    处理多个最大值的情况
    '''
    max=-1
    update=1
    for each in dict:
        if dict[each]<max: continue
        if dict[each]>max: update=1
        elif dict[each]==max: update=0
        max_name=each
        max=dict[each]
    if update==0: return 'none'#有多个最大值
    else: return max_name

def TimeModel(data,model,emotion):
    '''
    计算时间模式
    model是时间间隔秒数的整数
    返回时间区间和频数的字典
    '''
    res={}
    for line in tqdm(data):
        if line['emotion']!=emotion:#匹配情绪
            continue
        span=int(int(line['time'])/model)
        t_struct=time.localtime(span)
        start=str(time.strftime("%Y-%m-%d %H:%M:%S",t_struct))
        span+=model
        t_struct=time.localtime(span)
        end=str(time.strftime("%Y-%m-%d %H:%M:%S",t_struct))
        key_name=start+'~'+end
        if key_name not in res:#加入新区间
            res.update({key_name:1})
        else:#已有区间
            res[key_name]+=1
    return res

def DistanceRate(data,centre,distance_standard,emotion):
    '''
    计算data中与centre欧式距离不大于distance的评论中,情绪为emotion的比例
    '''
    count_in=0
    count_correct=0
    for line in data:
        dis=distance.euclidean(np.array(line['location']), np.array(centre))
        if dis<=distance_standard:
            count_in+=1
            if line['emotion']==emotion:
                count_correct+=1
    if count_in==0:
        return -1.0
    return count_correct/count_in

def main():
    '''
    测试函数
    '''
    file_path='D:\Project\Python\week3weibo\weibo.txt'#评论文件路径
    dict_path='D:\Project\Python\week3weibo\stopwords_list.txt'#分词词典路径
    emotion_lexicon_root='D:\Project\Python\week3weibo\emotion_lexicon'#情绪词典所在文件夹路径
    model=1*7*24*60*60
    centre=[39.000000,116.000000]#中心位置
    distance=1#半径
    data=ReadData(file_path)
    data=CleanData(data)
    data=CutText(data,dict_path)
    Emotion=LoadEmotion(emotion_lexicon_root)
    data=Emotion(data)
    time_model=TimeModel(data,model,'sadness')
    rate=DistanceRate(data,centre,distance,'joy')
    print(time_model)
    print('rate =',rate)

if __name__=='__main__':
    main()