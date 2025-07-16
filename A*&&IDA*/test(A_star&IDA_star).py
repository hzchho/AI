import copy
import heapq
import time
filename=input('Please enter one filename:')
file=open(filename)
Open=[]
Closed=set()
matrix=[]
way=[]#改变的方块
path=[]#路径
#定义一个正确位置的矩阵
correct_matrix=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]]
for i in range(4):
    alist=file.readline().split()
    matrix.append(alist)
#转化成数字
for i in range(4):
    for j in range(4):
        matrix[i][j]=int(matrix[i][j])

print("Choose the function to search:\n1.曼哈顿\n2.未复原方块数")
op=int(input())
#存储节点
class node:
    def __init__(self,g,h,matrix):
        self.g=g
        self.h=h
        self.f=self.g+self.h
        self.matrix=matrix
        self.prev=None
        self.changenum=0
    
    #heapq的比较函数
    def __lt__(self,Node): 
        if self.f==Node.f:
            return self.g<Node.g
        return self.f<Node.f
#A*算法
#以曼哈顿距离作为启发式函数的参考
#获取数字在矩阵中的位置（x，y）
def get_index(num,matrix):
    for i in range(4):
        for j in range(4):
            if matrix[i][j]==num:
                return i,j
#h(x)            
def h(matrix1,matrix2):#第一个为当前矩阵，第二个为目标矩阵
    eva_value=0
    if op==1:
        for i in range(4):
            for j in range(4):
                current_value=matrix1[i][j]
                if current_value==0:
                    continue
                target_x,target_y=get_index(current_value,matrix2)
                eva_value=eva_value+abs(target_x-i)+abs(target_y-j)
                
    elif op==2:
        for i in range(4):
            for j in range(4):
                if matrix1[i][j]!=0 and matrix1[i][j]!=matrix2[i][j]:
                    eva_value+=1
                
    return eva_value
#判断当前矩阵是否在已扩展的矩阵列表中
def find_end(matrix,visitlist):
    for x in visitlist:
        if matrix==x:
            return True
    
    return False
#从当前的节点回溯找到每次变动的节点
def find_way(state):
    if state.prev!=None and state.changenum!=0:
        way.append(state.changenum)
        path.append(state.matrix)
        find_way(state.prev)
#A*算法    
def A_star(matrix,count):
    Open=[]#待扩展列表
    Closed=set()#已扩展列表
    moves=[(0,1),(0,-1),(1,0),(-1,0)]#移动的方向
    root=node(0,h(matrix,correct_matrix),matrix)
    Open.append(root)
    #把扩展列表转化成堆
    heapq.heapify(Open)
    #将当前节点的矩阵转化成字符串对应的哈希值加入已扩展列表
    Closed.add(hash(str(root.matrix)))
    while len(Open)!=0:
        cur_state=heapq.heappop(Open)#取出堆中最小元素进行扩展
        count=count+1#扩展次数
        #结束条件1
        if cur_state.matrix==correct_matrix:
            find_way(cur_state)
            return count
        #结束条件2：
        if count>20000000:
            print("Too many expansion nodes(more than 2000w)!")
            return count
        #获取0的位置，即空格位
        idx,idy=get_index(0,cur_state.matrix)
        for i,j in moves:
            idx_=idx+i
            idy_=idy+j
            if 0<=idx_<4 and 0<=idy_<4:
                tmp_matrix=copy.deepcopy(cur_state.matrix)
                tmp_matrix[idx][idy],tmp_matrix[idx_][idy_]=tmp_matrix[idx_][idy_],tmp_matrix[idx][idy]
                hash_val=hash(str(tmp_matrix))
                #判断当前节点扩展出的节点是否已扩展
                if hash_val in Closed:
                    continue
                Closed.add(hash_val)
                next_state=node(cur_state.g+1,h(tmp_matrix,correct_matrix),tmp_matrix)
                next_state.prev=cur_state
                next_state.changenum=tmp_matrix[idx][idy]
                #将下一节点入堆
                heapq.heappush(Open,next_state)
    
    return count
#IDA*算法的递归步骤
def IDA_search(g,depth,count):
    cur_state=Open[-1]#此处的state是矩阵
    count+=1
    #当前节点的f(x)值
    f=g+h(cur_state,correct_matrix)
    #判断是否进行扩展
    if f>depth:
        return f,count
    if cur_state==correct_matrix:
        return 0,count
    
    mincost=90 #最远的曼哈顿距离不会大于每个都在对角线相反位置，即15*6=90
    moves=[(1,0),(-1,0),(0,1),(0,-1)]
    idx,idy=get_index(0,cur_state)
    new_list=[]
    for i,j in moves:
        idx_=idx+i
        idy_=idy+j
        if 0<=idx_<4 and 0<=idy_<4:
            tmp_matrix=copy.deepcopy(cur_state)
            tmp_matrix[idx][idy],tmp_matrix[idx_][idy_]=tmp_matrix[idx_][idy_],tmp_matrix[idx][idy]
            new_list.append(tmp_matrix)

    for next_state in new_list:
        if hash(str(next_state)) in Closed:
            continue
        Open.append(next_state)
        way.append(next_state[idx][idy])
        path.append(next_state)
        Closed.add(hash(str(next_state)))
        t,count=IDA_search(g+1,depth,count)
        
        if t==0:
            return 0,count
        if t<mincost:
            mincost=t
        #如果没有从当前节点的子节点中找到结果，就回退    
        Open.pop()
        way.pop()
        path.pop()
        Closed.remove(hash(str(next_state)))
    
    return mincost,count
#IDA*算法    
def IDA_star(matrix,count):
    depth=h(matrix,correct_matrix)
    Open.append(matrix)
    Closed=set()
    Closed.add(hash(str(matrix)))
    
    while(1):
        result,count=IDA_search(0,depth,count)
        if count>20000000:
            print("Too many expansion nodes(more than 2000w)!")
            return count
        if result==0:
            return count
        if result==-1:
            print('Failed to find solution!')
            return count
        depth=result
    

expand_num=0
time1=time.time()
print("Choose a function to run:\n1.A*\n2.IDA*")
op2=int(input())
if op2==1:
    print("A*:")
    expand_num=A_star(matrix,expand_num)
    way=way[::-1]
    path=path[::-1]
elif op2==2:
    print("IDA*:")
    expand_num=IDA_star(matrix,expand_num)
print("The num of node expanded are:",expand_num)
print("A optional solution is:",len(way))
print("The way of task are:")
for i in way:
    print(i,end=' ')
for j in range(len(path)):
    print("\nNo.",j+1,":\n",path[j],end='')
time2=time.time()
print('')
print("Used Time %f" %(time2-time1), "s")