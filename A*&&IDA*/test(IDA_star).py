import copy
import heapq
import time
filename=input('Please enter one filename:')
file=open(filename)
matrix=[]
Open=[]#待扩展列表:栈
Closed=[]#已扩展列表
way=[]#路径
#定义一个正确位置的矩阵
correct_matrix=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]]
for i in range(4):
    alist=file.readline().split()
    matrix.append(alist)
#转化成数字
for i in range(4):
    for j in range(4):
        matrix[i][j]=int(matrix[i][j])

class node:
    def __init__(self,g,h,matrix):
        self.g=g
        self.h=h
        self.f=self.g+self.h
        self.matrix=matrix
        self.prev=None
        self.changenum=0
         
def custom(a):
    return a.f      
#获取数字在矩阵中的位置（x，y）
def get_index(num,matrix):
    for i in range(4):
        for j in range(4):
            if matrix[i][j]==num:
                return i,j

#h(x):在A*算法中已经验证过曼哈顿距离作为估价函数效率更高，这里不做选择    
def h(matrix1,matrix2):#第一个为当前矩阵，第二个为目标矩阵
    eva_value=0
    for i in range(4):
        for j in range(4):
            current_value=matrix1[i][j]
            if current_value==0:
                continue
            target_x,target_y=get_index(current_value,matrix2)
            eva_value=eva_value+abs(target_x-i)+abs(target_y-j)
    
    return eva_value
#判断列表中是否有某个扩展的矩阵
def find_end(matrix,visitlist):
    for x in visitlist:
        if matrix==x:
            return True
    
    return False
#找到能找到结果的路径
def find_way(state):
    if state.prev!=None and state.changenum!=0:
        way.append(state.changenum)
        find_way(state.prev)
        
def IDA_star(matrix,count):
    moves=[(1,0),(-1,0),(0,1),(0,-1)]
    root=node(0,h(matrix,correct_matrix),matrix)
    Open.append(root)
    depth=root.f
    #把扩展列表转化成堆
    Closed.append(root.matrix)
    while(len(Open)!=0):
        cur_state=Open.pop()
        depth=depth+1
        count=count+1
        if cur_state.matrix==correct_matrix:
            find_way(cur_state)
            return count
        
        idx,idy=get_index(0,cur_state.matrix)
        for i,j in moves:
            idx_=idx+i
            idy_=idy+j
            if 0<=idx_<4 and 0<=idy_<4:
                tmp_matrix=copy.deepcopy(cur_state.matrix)
                tmp_matrix[idx][idy],tmp_matrix[idx_][idy_]=tmp_matrix[idx_][idy_],tmp_matrix[idx][idy]
                if h(tmp_matrix,correct_matrix)+cur_state.g+1>depth:
                    continue
                else:
                    if find_end(tmp_matrix,Closed):
                        continue
                    Closed.append(tmp_matrix)
                    next_state=node(cur_state.g+1,h(tmp_matrix,correct_matrix),tmp_matrix)
                    next_state.prev=cur_state
                    next_state.changenum=tmp_matrix[idx][idy]
                    Open.append(next_state)
    
    return count
            
    
expand_num=0#扩展的节点个数
time1=time.time()
expand_num=IDA_star(matrix,expand_num)
print("The num of node expanded are:",expand_num)
print("A optional solution is:",len(way))
way=way[::-1]
print("The way of task are:")
for i in way:
    print(i,end=' ')
time2=time.time()
print('')
print("Used Time %f" %(time2-time1), "s")