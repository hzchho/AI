import numpy as np
import matplotlib.pyplot as plt
import ast
import math
import random
filename="kmeans_data.csv"
index=[]
k=3
with open(filename,'r') as file:
    if_read=0
    for line in file:
        if if_read==0:
            if_read=1
            continue
        data=line.replace(',',' ').split()
        x_=ast.literal_eval(data[0])
        y_=ast.literal_eval(data[1])
        index.append((x_,y_))
#计算两个点之间的欧氏距离
def distance(point1,point2):
    return math.sqrt(sum([(a-b)**2 for a,b in zip(point1,point2)]))
#第一次kmeans，生成和分配
def kmeans(data,k):
    #随机生成k个聚类中心
    main_point=random.sample(data,k)
    l=len(main_point)
    
    unit=[[] for _ in range(l)]
    #将每个点归到与自己最近的聚类中心的聚类中
    for point in data:
        dis=[distance(point,a) for a in main_point]
        #找到与当前点距离最短的聚类中心在main_point列表对应的id
        main_point_id=dis.index(min(dis))
        unit[main_point_id].append(point)
    
    return main_point,unit
    
def refresh_kmeans(old_main_point,old_unit,data):
    new_main_point=[]
    end=0
    #由每个聚类的平均值作为聚类中心
    for u in old_unit:
        if len(u)==0:
            new_main_point.append(random.sample(index,1)[0])
            continue
        
        new_x=sum([a[0] for a in u])/len(u)
        new_y=sum([a[1] for a in u])/len(u)
        new_main_point.append((new_x,new_y))
    
    new_unit=[[] for _ in range(len(new_main_point))]
    #更新新的聚类
    for point in data:
        dis=[distance(point,a) for a in new_main_point]
        main_point_id=dis.index(min(dis))
        new_unit[main_point_id].append(point)
    #求新的聚类中心和旧的聚类中心的误差，如果收敛则即可结束    
    error=sum([distance(a,b) for a,b in zip(old_main_point,new_main_point)])
    if error<=0.00000001:
        end=1
    
    return new_main_point,new_unit,end
    

if __name__=="__main__":
    main_point,unit=kmeans(index,k)
    count=0
    if_end=0
    #进行多次更新和分配
    while if_end==0:
        main_point,unit,if_end=refresh_kmeans(main_point,unit,index)
        count+=1
        print("第",count,"次更新")
    
    colors=['r','g','b','y','c','m']
    for i,u in enumerate(unit):
        points = np.array(u)
        plt.scatter(points[:,0],points[:,1],c=colors[i],label=f'Cluster {i+1}')
    
    p=np.array(main_point)
    plt.scatter(p[:,0],p[:,0],c='black',label=f'Main Point')    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-Means Cluster')
    plt.legend()
    plt.show()
