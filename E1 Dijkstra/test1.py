m,n=map(int,input().split())
distance=[]
al={'a':0,'b':1,'c':2,'d':3,'e':4,'z':5}#将节点用序号表示
MAX=float('inf')  #定义最大值
def Dijkstra(arr,start,end):
    l=len(arr)
    start_id=al[start]  #将起始节点和终点替换成序号
    end_id=al[end]
    least=[MAX]*n   #距离矩阵
    least[start_id]=0
    v=[0]*n  #记录是否访问过
    path=[]
    #求最短路径的长度
    for i in range(l):
        min_num=MAX
        min_id=-1
        for index in range(l):
            #如果当前节点未访问过且小于最小值，更新最小值以及对应节点序号
            if v[index]==0 and least[index]<min_num:
                min_num=least[index]
                min_id=index
        #如果需要没更新则跳过后续更新路径步骤
        if min_id==-1:
            break
        v[min_id]=1 #记当前节点已访问
        
        for j in range(l):
            if arr[min_id][j]>0:
                #更新到该点的最短路径
                least[j]=min(least[j],least[min_id]+arr[min_id][j])
    
    #确认最短路径（从后往前，每一步都可以唯一确认）
    cur=end_id
    while cur!=start_id:
        path.insert(0,cur)#前插法
        for k in range(l):
            if arr[cur][k]>0 and least[cur]==least[k]+arr[k][cur]:
                cur=k
                break
    path.insert(0,start_id)
    #将数字转成节点名称
    for i in range(len(path)):
        for k,v in al.items():
            if v==path[i]:
                path[i]=k 
    print("The shortest path bewteen",start,"and",end,"are:",path)
    print("The distance is",least[end_id])
        
#读取存储图节点的文件   
file=open("data.txt") 
#创建距离矩阵
for i in range(m):
    tmp=[]
    for j in range(m):
        if i==j:
            tmp.append(0)
        else:
            tmp.append(MAX)#MAX表示不可达
            
    distance.append(tmp)
#输入图节点
for i in range(n):
    #fro,bac,weigh=map(str,input().split())
    line=file.readline()
    #从每一行中获取数据
    fro,bac,weigh=line.split()
    weigh=int(weigh)
    a=al[fro]
    b=al[bac]
    distance[a][b]=weigh
    distance[b][a]=weigh

for i in range(m):
    print(distance[i])
start,end=map(str,input().split())
while start!='x' and end!='x' :
    Dijkstra(distance,start,end)
    start,end=map(str,input().split())

print("End")
#参考文献：
#https://blog.csdn.net/qq_62789540/article/details/126044970?ops_request_misc=&request_id=&biz_id=102&utm_term=%E8%BF%AA%E6%9D%B0%E6%96%AF%E7%89%B9%E6%8B%89%E7%AE%97%E6%B3%95%E6%B1%82%E6%9C%80%E7%9F%AD%E8%B7%AF%E5%BE%84python&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-5-126044970.nonecase&spm=1018.2226.3001.4187
#https://blog.csdn.net/u014453898/article/details/86506981?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170930827316800225512513%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170930827316800225512513&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-86506981-null-null.142^v99^control&utm_term=%E8%BF%AA%E6%9D%B0%E6%96%AF%E7%89%B9%E6%8B%89%E7%AE%97%E6%B3%95%E6%B1%82%E6%9C%80%E7%9F%AD%E8%B7%AF%E5%BE%84python&spm=1018.2226.3001.4187




