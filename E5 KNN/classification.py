import math
from collections import Counter
import time
#k-NN分类器
train_file="code/train.txt"
test_file="code/test.txt"
#创建词向量表        
def create_array(filename):
    document=0
    with open(filename,"r",encoding='utf-8') as file:
        passline=1;
        data=[]
        for line in file:
            document+=1
            if passline==1:
                passline=0
                continue
            
            part=line.strip().split()
            for danci in part[3::]:
                data.append(danci)
                
    data1=set(data)
    words={word:0 for i,word in enumerate(data1)}#words为词向量表和包含该词的文档数
    return words,document
'''emotionid对应的emotion
0 anger
1 disgust
2 fear
3 joy
4 sad
5 surprise
'''  
words,N=create_array(train_file)
k=int(math.sqrt(N))#确定k的值
#train
#read_data
#生成测试集的对照表
def read_data(filename):
    table={}
    TF={}
    with open(filename,"r",encoding='utf-8') as file:
        passline=1
        for line in file:
            if passline==1:
                passline=0
                continue
            
            part=line.strip().split()
            data=part[3::]
            total=len(data)
            tmp=""
            freq=[]
            for key in words.keys():
                appearance=0
                count=0
                for i in data:
                    if i==key:
                        appearance=1
                        count+=1
                        break
                    
                if appearance==1:
                    tmp+="1"
                    words[key]+=1
                else:
                    tmp+="0"
                    
                freq.append(count/total)
            
            table[tmp]=int(part[1])
            TF[tmp]=freq
    
    IDF=[]
    for value in words.values():
        tmp=math.log(N/(value+1))
        IDF.append(tmp)  
    #IDF是nx1维向量，TF为Nxn，n为词向量表的词数      
    return table,TF,IDF
#计算两个文档向量表之间的距离
def distance(list1,list2,op):
    f=0
    if op==0:#manhatan
        f=sum([abs(a-b) for a,b in zip(list1,list2)]) 
    elif op==1:#欧式距离
        f=math.sqrt(sum([(a-b)**2 for a,b in zip(list1,list2)]))
    elif op==2:#无穷范数
        f=max([abs(a-b) for a,b in zip(list1,list2)])
    
    return f
#求列表中出现次数最多的元素：即在k近邻中找众数
def most_common_element(lst):
    # 使用Counter统计列表中每个元素的出现次数
    counts=Counter(lst)
    # 使用Counter的most_common()方法找到出现次数最多的元素及其出现次数
    most_common=counts.most_common(1)
    # 返回出现次数最多的元素
    return most_common[0][0]
#求预测标签        
def predicition(data,train_table,train_TF_IDF,train_IDF,real_label,op):
    pred_label=0
    TF_IDF2=[a*b for a,b in zip(data,train_IDF)]
    store_list=[]
    for key,value in train_TF_IDF.items():
        dis=distance(TF_IDF2,value,op)
        label=train_table[key]
        store_list.append((dis,label))
        
    store_list=sorted(store_list, key=lambda x: x[0])[:k:]
    pred_label=most_common_element(store_list)
    #print("pred_label:",pred_label[1],"real_label:",real_label)
    return (pred_label[1]==real_label)
#计算准确率
def Accuracy(filename,train_table,train_TF_IDF,train_IDF,op):
    result=[]
    with open(filename,"r",encoding='utf-8') as file:
        passline=1
        for line in file:
            if passline==1:
                passline=0
                continue
            
            part=line.strip().split()
            data=part[3::]
            total=len(data)
            freq=[]
            for key in words.keys():
                count=0
                for i in data:
                    if i==key:
                        count+=1
                        break
                    
                freq.append(count/total)
            
            tmp=predicition(freq,train_table,train_TF_IDF,train_IDF,int(part[1]),op)
            result.append(tmp)
            
    accuracy=result.count(1)/len(result)*100.0
    return accuracy

time1=time.time()
train_table,train_TF,train_IDF=read_data(train_file)
TF_IDF={}
for key,value in train_TF.items():
    tmp=[a*b for a,b in zip(value,train_IDF)]
    TF_IDF[key]=tmp
     
op=2
result=Accuracy(test_file,train_table,TF_IDF,train_IDF,op)
time2=time.time()
using_function=""
if op==0:#manhatan
    using_function="Manhatan Distance"
elif op==1:#欧式距离
    using_function="Euclidean Distance"
elif op==2:#无穷范数
    using_function="Maximum Norm"
print("Using function:",using_function)
print("K:",k,"N:",N)
print("The accuracy is:",result,"%")
print("Used time:",time2-time1,"s")