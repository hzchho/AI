import pandas as pd
import matplotlib.pyplot as plt
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
import numpy as np
import time

#读入数据以及预处理
df=pd.read_csv('data/train_40.tsv',sep='\t',header=0,quoting=csv.QUOTE_NONE)
df2=pd.read_csv('data/dev_40.tsv',sep='\t',header=0,quoting=csv.QUOTE_NONE)
#去除重复值
df=df.drop_duplicates()
#数据类型转换，将标签转换为数值
label_map={'entailment':1,'not_entailment':0}
df['label']=df['label'].map(label_map)
df2['label']=df2['label'].map(label_map)
#转化成小写
df['question']=df['question'].str.lower()
df['sentence']=df['sentence'].str.lower()
df2['question']=df2['question'].str.lower()
df2['sentence']=df2['sentence'].str.lower()
#将问题和回答结合
df['combined']=(df['question']+df['sentence']).astype(str)
df2['combined']=(df2['question']+df2['sentence']).astype(str)
#count_ones=np.sum(df['label'].values==1)
#print(count_ones,len(df['label'].values))

#自定义数据集类
class TextDataset(Dataset):
    def __init__(self,dataframe,tokenizer,glove):
        """
        初始化 TextDataset 类。
        参数：
        - dataframe (DataFrame): 包含 'combined'（文本）和 'label' 列的 pandas DataFrame。
        - tokenizer: 用于将文本分词为标记的函数。
        - glove: GloVe 词向量模型（假设具有 `stoi` 表示词汇到索引的映射和 `vectors` 表示词向量的属性）。
        """
        self.df=dataframe
        self.tokenizer=tokenizer
        self.glove=glove
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        item=self.df.iloc[idx]#获取指定索引的 DataFrame行
        text=item['combined']
        label=item['label']
        text_vector=self.text_to_vector(text)
        return text_vector,label#返回元组
    
    def text_to_vector(self,text):
        tokens=self.tokenizer(text)#使用分词器分词
        vectors=[]
        for token in tokens:
            if token in self.glove.stoi:# 如果词在 GloVe 词汇表中
                vectors.append(self.glove.vectors[self.glove.stoi[token]])
            else:
                vectors.append(torch.zeros(100))#使用全0向量表示未知词
        # 保证返回一个有效的张量
        return torch.stack(vectors) if vectors else torch.zeros(1,100)

#处理批次数据
def collate_fn(batch):
    #解包批次数据，得到文本向量和标签的列表
    texts,labels=zip(*batch)
    #计算每个文本向量的长度
    lengths=torch.tensor([len(t) for t in texts])
    #对文本向量进行填充，使它们具有相同的长度
    texts_pad=pad_sequence(texts,batch_first=True)
    labels=torch.tensor(labels)
    
    return texts_pad,lengths,labels
#RNN模型
class RNN(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,output_dim,n_layers=1,bidirectional=False,dropout=0.5):
        """
        初始化 RNN 模型。
        参数：
        - embedding_dim (int): 词向量的维度。
        - hidden_dim (int): RNN 隐藏层的维度。
        - output_dim (int): 输出层的维度，通常等于类别的数量。
        - n_layers (int): RNN 层的数目，默认为 1。
        - bidirectional (bool): 是否使用双向 RNN,默认为 False。
        - dropout (float): Dropout 概率，默认为 0.5。
        """
        super(RNN,self).__init__()
        #双向LSTM
        self.rnn=nn.LSTM(embedding_dim,hidden_dim,num_layers=n_layers,bidirectional=bidirectional,batch_first=True,dropout=dropout)
        self.fc1=nn.Linear(hidden_dim*2 if bidirectional else hidden_dim,32)
        self.fc2=nn.Linear(32,output_dim)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x,lengths):
        #使用 pack_padded_sequence 对输入序列进行打包，以忽略填充部分
        #lengths.cpu()：涉及索引或特定功能时，将长度张量移到 CPU 上可以提高兼容性和稳定性
        packed_input=pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
        #将打包后的序列输入到 LSTM
        packed_output, (hidden, cell) = self.rnn(packed_input)
        #使用 pad_packed_sequence 将 LSTM 的输出解包，恢复到原始的 batch_size x sequence_length 形状
        output,_=pad_packed_sequence(packed_output,batch_first=True)
        #如果是双向 RNN，将两个方向的最后时刻的隐藏状态拼接起来
        if self.rnn.bidirectional:
            hidden=self.dropout(torch.cat((hidden[-2,:,:],hidden[-1,:,:]),dim=1))
        else:
            #如果是单向 RNN，只使用最后一个时刻的隐藏状态
            hidden=self.dropout(hidden[-1,:,:])
        #将隐藏状态传递给全连接层，得到最终的输出
        x=F.relu(self.fc1(hidden))
        return self.fc2(x)
#绘图    
def draw_picture(loss,correct):
    x=[i for i in range(len(loss))]
    plt.plot(x,loss,c='r',label='Loss')
    plt.plot(x,correct,c='b',label='Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Acc")
    plt.legend()
    plt.show()
    
#保存模型
def save_model(model,model_name):
    # 指定保存路径
    save_path=f"{model_name}.pth"
    # 创建模型字典
    model_dict={
        'model_state':model.state_dict(),
        'model_class':model.__class__.__name__
    }
    # 保存模型
    torch.save(model_dict,save_path)
    print(f"Model saved to {save_path}")

#主函数        
def main():
    #开始时间
    time1=time.time()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU是否可用
    #加载GloVe词向量
    glove=GloVe(name='6B',dim=100)
    tokenizer=get_tokenizer('basic_english')  
    #加载训练集  
    train_dataset=TextDataset(df,tokenizer,glove)
    train_dataloader=DataLoader(train_dataset,batch_size=256,collate_fn=collate_fn,shuffle=True)
    #初始化模型
    #双向LSTM
    model=RNN(embedding_dim=100,hidden_dim=128,output_dim=2,n_layers=2,bidirectional=True,dropout=0.5).to(device)
    #损失函数和优化器
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    #训练
    epochs=12
    loss_set=[]
    correct_set=[]
    #训练模式
    model.train()
    for epoch in range(epochs):
        total_loss=0
        correct=0
        time3=time.time()
        for texts_pad,lengths,labels in train_dataloader:
            texts_pad,lengths,labels=texts_pad.to(device),lengths.to(device),labels.to(device) #将数据移动到GPU上
            #优化器的梯度清零
            optimizer.zero_grad()
            #前向传播得到结果
            outputs=model(texts_pad,lengths)
            #求预测值
            _,predicted=torch.max(outputs,1)
            #累加正确的数据量
            correct+=(predicted==labels).sum().item()
            #求损失函数
            loss=criterion(outputs,labels)
            total_loss+=loss.item()
            #反向传播
            loss.backward()
            #计算梯度，更新优化器
            optimizer.step()
            
        loss_set.append(total_loss/len(train_dataloader))
        correct_set.append(correct/len(train_dataloader.dataset))
        time4=time.time()
        print(f"Epoch {epoch+1:<3} Loss: {total_loss/len(train_dataloader):>8.6f}  Acc: {correct*100/len(train_dataloader.dataset):5>.2f}%  Time: {time4-time3:>5.2f}sec")
    
    draw_picture(loss_set,correct_set)
    
    #模型评估
    #评估模式
    model.eval()
    #导入测试数据
    test_dataset=TextDataset(df2,tokenizer,glove)
    test_dataloader=DataLoader(test_dataset,batch_size=128,collate_fn=collate_fn,shuffle=True)
    correct=0
    total=0
    #语句块内的所有操作都不会计算梯度
    with torch.no_grad():
        for t,le,la in test_dataloader:
            #与训练过程相似
            t,le,la=t.to(device),le.to(device),la.to(device)
            output=model(t,le)
            _,pre=torch.max(output,1)
            total+=la.size(0)
            correct+=(pre==la).sum().item()
    #结束时间
    time2=time.time()    
    print(f"Accuracy: {correct /total*100:>5.2f}%   Total Time: {time2-time1:>6.2f}sec")
    #保存模型
    #save_model(model,"RNN_QNLI")
    
if __name__=='__main__':
    main()