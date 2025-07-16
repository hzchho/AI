import torch
from torch import nn
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
#数据路径
train_path="cnn_data/train"
test_path="cnn_data/test"
# 将一系列数据预处理步骤组合在一起
transform=transforms.Compose([
    #短边调整256，长边按比例缩放
    transforms.Resize(256),
    #裁剪成224*224
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #标准化为均值[0.485, 0.456, 0.406]，标准差[0.229, 0.224, 0.225]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 创建ImageFolder数据集实例
train_set=datasets.ImageFolder(root=train_path,transform=transform)
test_set=datasets.ImageFolder(root=test_path,transform=transform)
# 加载数据集
trainloader=torch.utils.data.DataLoader(train_set,batch_size=32,shuffle=True)#训练集大小为902,trainloader长度为29
testloader=torch.utils.data.DataLoader(test_set,batch_size=32,shuffle=False)#测试集大小为10
#定义卷积神经网络类
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
                #输入为3*224*224（channel x width x height）
                #根据计算公式：
                #width出=(width入-kernel_size+2*padding)/stride+1
                #height出=(height入-kernel_size+2*padding)/stride+1
                in_channels=3,
                #输出为16*226*226
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),#激活函数
            nn.MaxPool2d(kernel_size=2)#2x2的方块进行池化维度变为16*113*113
        )#
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,3,1,2),#16*113*123->32*115*115
            nn.ReLU(),
            nn.MaxPool2d(2)#32*115*115->32*57*57
        )
        self.fc1=nn.Linear(32*57*57,5)#32*57*57->5
        self.fc2=nn.Linear(256,32)
        self.fc3=nn.Linear(32,5)
        
    def forward(self,x):
        x=torch.relu(self.conv1(x))
        x=torch.relu(self.conv2(x))
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        #x=self.fc2(x)
        #x=self.fc3(x)
        return x

model=CNN()
#损失函数
Loss=nn.CrossEntropyLoss()
#优化函数
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)

#train
iteration=24
loss_set=[]
acc_set=[]
for epoch in range(iteration):
    #调用了model.train()表示进入训练模式，这会启用dropout和batch normalization等特殊行为
    model.train()
    time1=time.time()
    loss=0
    for batch_idx, (data,target) in enumerate(trainloader):
        optimizer.zero_grad()#清楚上一步的梯度
        out=model(data)#初始化
        loss=Loss(out,target)#计算损失函数
        loss_set.append(loss.item())
        loss.backward()#反向传播
        optimizer.step()#优化器更新
        if batch_idx==0 and epoch==0:
            print(f'Initial Loss: {loss.item():>8.6f}')
    
    #test
    #在测试时，调用了model.eval()表示进入测试模式，此时不会进行dropout和batch normalization等特殊行为
    model.eval()
    correct=0
    #为了防止跟踪历史记录（和使用内存），使用with torch.no_grad()封装
    with torch.no_grad():
        for data,target in testloader:
            out=model(data)
            pred=out.data.max(1,keepdim=True)[1]
            correct+=pred.eq(target.data.view_as(pred)).sum()
        
    acc_set.append(correct/len(testloader.dataset))
    time2=time.time()
    print(f'Train Epoch: [{epoch+1:<2d}/{iteration}] \tLoss: {loss.item():>10.6f} \tAccuracy: {correct*100/len(testloader.dataset):<5.2f}% \tUsed time: {time2-time1:>6.5f}s' )
#绘图            
x0=[i for i in range(iteration*len(trainloader))]
x1=[i for i in range(iteration)]
fig,ax=plt.subplots(1,2,figsize=(12,4))
ax[0].plot(x0,loss_set)
ax[0].set_xlabel("Epoch*data_num")
ax[0].set_ylabel("Loss")
ax[1].plot(x1,acc_set)
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Acc")   
plt.show()     
