def Search(board, EMPTY, BLACK, WHITE, isblack):
    # 目前 AI 的行为是随机落子，请实现 AlphaBetaSearch 函数后注释掉现在的 return 
    # 语句，让函数调用你实现的 alpha-beta 剪枝
    return AlphaBetaSearch(board, EMPTY, BLACK, WHITE, isblack)
    #return RandomSearch(board, EMPTY)

def RandomSearch(board, EMPTY):
    # AI 的占位行为，随机选择一个位置落子
    # 在实现 alpha-beta 剪枝中不需要使用
    from random import randint
    ROWS = len(board)
    x = randint(0, ROWS - 1)
    y = randint(0, ROWS - 1)
    while board[x][y] != EMPTY:
        x = randint(0, ROWS - 1)
        y = randint(0, ROWS - 1)
    return x, y, 1

def AlphaBetaSearch(board, EMPTY, BLACK, WHITE, isblack):
    '''
    ---------------参数---------------
    board       当前的局面，是 15×15 的二维 list，表示棋盘
    EMPTY       空格在 board 中的表示，默认为 -1
    BLACK       黑棋在 board 中的表示，默认为 1
    WHITE       白棋在 board 中的表示，默认为 0
    isblack     bool 变量，表示当前是否轮到黑子落子
    ---------------返回---------------
    x           落子的 x 坐标（行数/第一维）
    y           落子的 y 坐标（列数/第二维）
    alpha       本层的 alpha 值
    '''
    # 请修改此函数，实现 alpha-beta 剪枝
    # =============你的代码=============
    score_graph=[]
    cut_count=0
    for i in range(15):
        new_=[]
        for j in range(15):
            new_.append('xxxx')
        score_graph.append(new_)
        
    best_move=(-1,-1)
    best_score=-float('inf') if isblack else float('inf')
    alpha=-float('inf')
    beta=float('inf')
    depth=2
    for idx,idy in get_next_move_locations(board,EMPTY):
        next_board=copy.deepcopy(board)
        next_board[idx][idy]=BLACK if isblack else WHITE
        #tmp_board=get_mi_shape(next_board,idx,idy)
        #if evaluation(tmp_board,isblack)>10000:
        #    best_score=evaluation(tmp_board,isblack)
        #   best_move=(idx,idy)
        #    break
        score,cut_count=ab_search(next_board,EMPTY,BLACK,WHITE,not isblack,depth-1,alpha,beta,cut_count)
        score_graph[idx][idy]=score
        if (isblack and score>best_score) or (not isblack and score<best_score):
            best_score=score
            best_move=(idx,idy)
        
    for i in range(15):
        for j in range(15):
            print(str(score_graph[i][j]).rjust(7),end=' ')
        print('') 
    
    print("进行了：",cut_count,"次剪枝")    
    return best_move[0],best_move[1],best_score


import copy
def ab_search(board,EMPTY,BLACK,WHITE,isblack,depth,alpha,beta,cut_count):
    if depth==0:
        #next_board=get_mi_shape(board,idx_,idy_)
        return evaluation(board,not isblack),cut_count
    next_step=0
    if isblack==1:
        next_step=BLACK
        best_score=-float('inf')
        for idx,idy in get_next_move_locations(board,EMPTY):
            next_board=copy.deepcopy(board)
            next_board[idx][idy]=next_step
            score,cut_count=ab_search(next_board,EMPTY,BLACK,WHITE,not isblack,depth-1,alpha,beta,cut_count)
            best_score=max(best_score,score)
            alpha=max(alpha,best_score)
            if beta<=alpha:  # 发生剪枝
                cut_count+=1
                break
                
        return best_score,cut_count
            
    else:
        next_step=WHITE
        best_score=float('inf')
        for idx,idy in get_next_move_locations(board,EMPTY):
            next_board=copy.deepcopy(board)
            next_board[idx][idy]=next_step
            score,cut_count=ab_search(next_board,EMPTY,BLACK,WHITE,not isblack,depth-1,alpha,beta,cut_count)
            best_score=min(best_score,score)
            beta=min(alpha,best_score)
            if beta<=alpha:  # 发生剪枝
                cut_count+=1
                break
            
        return best_score,cut_count

# 你可能还需要定义评价函数或者别的什么
# =============你的代码=============
score_shape=[(1, (-1, 1, 1, -1, -1)),#活二
             (1, (-1, -1, 1, 1, -1)),
             (15, (0, 1, 1, -1, 1, -1)),#冲三
             (15, (-1, 1, 1, 1, 0)),
             (15, (0, 1, -1, 1, 1, -1)),
             (15, (1, 1, -1, -1, 1)),
             (15, (1, -1, 1, -1, 1)),
             (50, (-1, 1, 1, 1, -1)),#活三
             (50, (-1, 1, -1, 1, 1, -1)),
             (50, (0, 1, 1, 1, -1, 1)),#冲四
             (50, (1, 1, -1, 1, 1)),
             (50, (0, 1, 1, 1, 1, -1)),
             (250, (1, 0, 0, 0, -1)),#堵活三
             (250, (-1, 0, 1, 0, 0, -1)),
             (1000, (-1, 1, 1, 1, 1, -1)),#活四
             (10001, (1, 0, 0, 0, 0, 1)),#堵冲四
             (10001, (0, 0, 1, 0, 0)),
             (10001, (-1, 0, 0, 0, 1, 0)),
             (10001, (-1, 0, 0, 0, 0, 1)),
             (500000, (1, 1, 1, 1, 1))]#胜利
'''
             (-1, (-1, 0, 0, -1, -1)),#敌方活二
             (-1, (-1, -1, 0, 0, -1)),
             (-15, (1, 0, 0, -1, 0, -1)),#敌方冲三
             (-15, (-1, 0, 0, 0, 1)),
             (-15, (1, 0, -1, 0, 0, -1)),
             (-15, (0, 0, -1, -1, 0)),
             (-15, (0, -1, 0, -1, 0)),
             (-50, (-1, 0, 0, 0, -1)),#敌方活三
             (-50, (-1, 0, -1, 0, 0, -1)),
             (-50, (1, 0, 0, 0, -1, 0)),#敌方冲四
             (-50, (0, 0, -1, 0, 0)),
             (-50, (1, 0, 0, 0, 0, -1)),
             (-250, (0, 1, 1, 1, -1)),#敌方堵活三
             (-250, (-1, 1, 0, 1, 1, -1)),
             (-1000, (-1, 0, 0, 0, 0, -1)),#敌方活四
             (-10001, (0, 1, 1, 1, 1, 0)),#敌方堵活四
             (-10001, (1, 1, 0, 1, 1)),
             (-10001, (-1, 1, 1, 1, 0, 1)),
             (-10001, (-1, 1, 1, 1, 1, 0)),
             (-500000, (0, 0, 0, 0, 0))]
             #敌方胜利'''

           
def get_mi_shape(board,idx,idy):
    s=[]
    for i in range(15):
        new_=[]
        for j in range(15):
            new_.append(-1)
        s.append(new_)
    s[idx][idy]=board[idx][idy]
    for i in range(6):
        if idx-i>=0 and idy-i>=0:
            s[idx-i][idy-i]=board[idx-i][idy-i]
        if idx-i>=0 and idy+i<=14:
            s[idx-i][idy+i]=board[idx-i][idy+i]
        if idx+i<=14 and idy-i>=0:
            s[idx+i][idy-i]=board[idx+i][idy-i]
        if idx+i<=14 and idy+i<=14:
            s[idx+i][idy+i]=board[idx+i][idy+i]
        if idx-i>=0:
            s[idx-i][idy]=board[idx-i][idy]
        if idy-i>=0:
            s[idx][idy-i]=board[idx][idy-i]
        if idx+i<=14 :
            s[idx-i][idy]=board[idx-i][idy]
        if idy+i<=14:
            s[idx][idy+i]=board[idx][idy+i]
    return s

def evaluation(board,isblack):
    sum_score=0
    for score,pattern in score_shape:
        count=count_pattern(board,pattern)
        #if (isblack and score>0) or (not isblack and score<0):
        sum_score+=count*score    
    
    return sum_score    
                        
                
# 以下为编写搜索和评价函数时可能会用到的函数，请看情况使用、修改和优化
# =============辅助函数=============

def _coordinate_priority(coordinate):
    x, y = coordinate[0], coordinate[1]
    return x * 15 + y

def get_successors(board, color, priority=_coordinate_priority, EMPTY=-1):
    '''
    返回当前状态的所有后继（默认按坐标顺序从左往右，从上往下）
    ---------------参数---------------
    board       当前的局面，是 15×15 的二维 list，表示棋盘
    color       当前轮到的颜色
    EMPTY       空格在 board 中的表示，默认为 -1
    priority    判断落子坐标优先级的函数（结果为小的优先）
    ---------------返回---------------
    一个生成器，每次迭代返回一个的后继状态 (x, y, next_board)
        x           落子的 x 坐标（行数/第一维）
        y           落子的 y 坐标（列数/第二维）
        next_board  后继棋盘
    '''
    # 注意：生成器返回的所有 next_board 是同一个 list！
    from copy import deepcopy
    next_board = deepcopy(board)
    ROWS = len(board)
    idx_list = [(x, y) for x in range(15) for y in range(15)]
    idx_list.sort(key=priority)
    print(idx_list)
    for x, y in idx_list:
        if board[x][y] == EMPTY:
            next_board[x][y] = color
            yield (x, y, next_board)
            next_board[x][y] = EMPTY


# 这是使用 successors 函数的一个例子，打印所有后继棋盘
def _test_print_successors():
    '''
    棋盘：
      0 y 1   2
    0 1---+---1
    x |   |   |
    1 +---0---0
      |   |   |
    2 +---+---1
    本步轮到 1 下
    '''
    board = [
        [ 1, -1,  1],
        [-1,  0,  0],
        [-1, -1,  1]]
    EMPTY = -1
    next_states = get_successors(board, 1)
    for x, y, state in next_states:
        print(x, y, state)
    # 输出：
    # 0 1 [[1, 1, 1], [-1, 0, 0], [-1, -1, 1]]
    # 1 0 [[1, -1, 1], [1, 0, 0], [-1, -1, 1]]
    # 2 0 [[1, -1, 1], [-1, 0, 0], [1, -1, 1]]
    # 2 1 [[1, -1, 1], [-1, 0, 0], [-1, 1, 1]]

def get_next_move_locations(board, EMPTY=-1):
    '''
    获取下一步的所有可能落子位置
    ---------------参数---------------
    board       当前的局面，是 15×15 的二维 list，表示棋盘
    EMPTY       空格在 board 中的表示，默认为 -1
    ---------------返回---------------
    一个由 tuple 组成的 list，每个 tuple 代表一个可下的坐标
    '''
    next_move_locations = []
    ROWS = len(board)
    for x in range(ROWS):
        for y in range(ROWS):
            if board[x][y] == EMPTY:
                next_move_locations.append((x,y))
    
    middle=len(next_move_locations)//2
    next_move_locations=next_move_locations[middle:]+next_move_locations[:middle]
    return next_move_locations

def get_pattern_locations(board, pattern):
    '''
    获取给定的棋子排列所在的位置
    ---------------参数---------------
    board       当前的局面，是 15×15 的二维 list，表示棋盘
    pattern     代表需要找的排列的 tuple
    ---------------返回---------------
    一个由 tuple 组成的 list，每个 tuple 代表在棋盘中找到的一个棋子排列
        tuple 的第 0 维     棋子排列的初始 x 坐标（行数/第一维）
        tuple 的第 1 维     棋子排列的初始 y 坐标（列数/第二维）
        tuple 的第 2 维     棋子排列的方向，0 为向下，1 为向右，2 为右下，3 为左下；
                            仅对不对称排列：4 为向上，5 为向左，6 为左上，7 为右上；
                            仅对长度为 1 的排列：方向默认为 0
    ---------------示例---------------
    对于以下的 board（W 为白子，B为黑子）
      0 y 1   2   3   4   ...
    0 +---W---+---+---+-- ...
    x |   |   |   |   |   ...
    1 +---+---B---+---+-- ...
      |   |   |   |   |   ...
    2 +---+---+---W---+-- ...
      |   |   |   |   |   ...
    3 +---+---+---+---+-- ...
      |   |   |   |   |   ...
    ...
    和要找的 pattern (WHITE, BLACK, WHITE)：
    函数输出的 list 会包含 (0, 1, 2) 这一元组，代表在 (0, 1) 的向右下方向找到了
    一个对应 pattern 的棋子排列。
    '''
    ROWS = len(board)
    DIRE = [(1, 0), (0, 1), (1, 1), (1, -1)]
    pattern_list = []
    palindrome = True if tuple(reversed(pattern)) == pattern else False
    for x in range(ROWS):
        for y in range(ROWS):
            if pattern[0] == board[x][y]:
                if len(pattern) == 1:
                    pattern_list.append((x, y, 0))
                else:
                    for dire_flag, dire in enumerate(DIRE):
                        if _check_pattern(board, ROWS, x, y, pattern, dire[0], dire[1]):
                            pattern_list.append((x, y, dire_flag))
                    if not palindrome:
                        for dire_flag, dire in enumerate(DIRE):
                            if _check_pattern(board, ROWS, x, y, pattern, -dire[0], -dire[1]):
                                pattern_list.append((x, y, dire_flag + 4))
    return pattern_list

# get_pattern_locations 调用的函数
def _check_pattern(board, ROWS, x, y, pattern, dx, dy):
    for goal in pattern[1:]:
        x, y = x + dx, y + dy
        if x < 0 or y < 0 or x >= ROWS or y >= ROWS or board[x][y] != goal:
            return False
    return True

def count_pattern(board, pattern):
    # 获取给定的棋子排列的个数
    return len(get_pattern_locations(board, pattern))

def is_win(board, color, EMPTY=-1):
    # 检查在当前 board 中 color 是否胜利
    pattern1 = (color, color, color, color, color)          # 检查五子相连
    pattern2 = (EMPTY, color, color, color, color, EMPTY)   # 检查「活四」
    return count_pattern(board, pattern1) + count_pattern(board, pattern2) > 0

# 这是使用以上函数的一个例子
def _test_find_pattern():
    '''
    棋盘：
      0 y 1   2   3   4   5
    0 1---+---1---+---+---+
    x |   |   |   |   |   |
    1 +---0---0---0---0---+ ... 此行有 0 的「活四」
      |   |   |   |   |   |
    2 +---+---1---+---+---1
      |   |   |   |   |   |
    3 +---+---+---+---0---+
      |   |   |   |   |   |
    4 +---+---+---1---0---1
      |   |   |   |   |   |
    5 +---+---+---+---+---+
    '''
    board = [
        [ 1, -1,  1, -1, -1, -1],
        [-1,  0,  0,  0,  0, -1],
        [-1, -1,  1, -1, -1,  1],
        [-1, -1, -1, -1,  0, -1],
        [-1, -1, -1,  1,  0,  1],
        [-1, -1, -1, -1, -1, -1]]
    pattern = (1, 0, 1)
    pattern_list = get_pattern_locations(board, pattern)
    assert pattern_list == [(0, 0, 2), (0, 2, 0), (2, 5, 3), (4, 3, 1)]
        # (0, 0) 处有向右下的 pattern
        # (0, 2) 处有向下方的 pattern
        # (2, 5) 处有向左下的 pattern
        # (4, 3) 处有向右方的 pattern
    assert count_pattern(board, (1,)) == 6
        # 6 个 1
    assert count_pattern(board, (1, 0)) == 13
        # [(0, 0, 2), (0, 2, 0), (0, 2, 2), (0, 2, 3), (2, 2, 4), 
        #  (2, 2, 6), (2, 2, 7), (2, 5, 3), (2, 5, 6), (4, 3, 1), 
        #  (4, 3, 7), (4, 5, 5), (4, 5, 6)]
    assert is_win(board, 1) == False
        # 1 没有达到胜利条件
    assert is_win(board, 0) == True
        # 0 有「活四」，胜利
