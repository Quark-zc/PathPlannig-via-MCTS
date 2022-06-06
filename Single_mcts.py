"""
5.14 单机器人MCTS覆盖算法仿真

"""
import sys

from treelib import Tree, Node
import random
import math
import matplotlib.pyplot as plt
import time


'''
1.主函数部分
'''

# todo 覆盖过的栅格标志位设置为0.5， 未覆盖的为0 ， 障碍物为1
# todo 算法效果差主要是随机策略设计的有问题

def tree_update(node_id, avail_move):  # 更新树:输入父节点id和可行动作集合,随机扩展一个子节点并返回其id

    choice = random.choice(avail_move)
    while choice == 0:
        choice = random.choice(avail_move)

    x = tree.get_node(nid=node_id).data[0][0]
    y = tree.get_node(nid=node_id).data[0][1]

    sub_node_id = 0

    if (choice == 1):
        sub_node = Node(tag='straight', data=[(x - 1, y), 0, 10e-5, 1,1e5])
        tree.add_node(sub_node, parent=node_id)
        sub_node_id = sub_node.identifier
        coveraged.append((x - 1, y))

    if (choice == 2):
        sub_node = Node(tag='back', data=[(x + 1, y), 0, 10e-5, 2,1e5])
        tree.add_node(sub_node, parent=node_id)
        sub_node_id = sub_node.identifier
        coveraged.append(((x + 1, y)))

    if (choice == 3):
        sub_node = Node(tag='turn left', data=[(x, y - 1), 0, 10e-5, 3,1e5])
        tree.add_node(sub_node, parent=node_id)
        sub_node_id = sub_node.identifier
        coveraged.append((x, y - 1))

    if (choice == 4):
        sub_node = Node(tag='turn right', data=[(x, y + 1), 0, 10e-5, 4,1e5])
        tree.add_node(sub_node, parent=node_id)
        sub_node_id = sub_node.identifier
        coveraged.append((x, y + 1))

    return (sub_node_id)

def default_policy(node_id):    # 根据内螺旋策略向下选择一个节点进行仿真，遇障碍物或者已覆盖栅格右转

    # todo move = tree.parent(node_id).data[3]
    move = tree.get_node(node_id).data[3]
    all_moves =  get_moni_move(node_id)
    if all_moves == [0,0,0,0]:  # 陷入死区
        all_moves = get_simulation_move(node_id)

    if move in all_moves:
        sub_id = tree_update(node_id, [move])
    else:
        move = random.choice(all_moves)
        while move == 0:
            move = random.choice(all_moves)
        sub_id = tree_update(node_id,[move])

    return sub_id

def is_end(leaf_id):     # 判断当前叶子节点表示状态是否为结束局面

    pass


def get_expand_move(node_id):  # 获得可扩展的操作

    children = tree.children(nid=node_id)
    expand_avail_move = [1, 2, 3, 4]

    for i in range(len(children)):  # 将已经扩展的节点置零
        if children[i].tag == 'straight':
            expand_avail_move[0] = 0
        elif children[i].tag == 'back':
            expand_avail_move[1] = 0
        elif children[i].tag == 'turn left':
            expand_avail_move[2] = 0
        elif children[i].tag == 'turn right':
            expand_avail_move[3] = 0

    x = tree.get_node(nid=node_id).data[0][0]  # 将不在通行区域的操作置零
    y = tree.get_node(nid=node_id).data[0][1]
    if (((x - 1, y) in pass_point) == 0):
        expand_avail_move[0] = 0
    if (((x + 1, y) in pass_point) == 0):
        expand_avail_move[1] = 0
    if (((x, y - 1) in pass_point) == 0):
        expand_avail_move[2] = 0
    if (((x, y + 1) in pass_point) == 0):
        expand_avail_move[3] = 0

    return (expand_avail_move)


def get_simulation_move(node_id):  # 获得可模拟的操作

    simulation_avail_move = [1, 2, 3, 4]
    x = tree.get_node(nid=node_id).data[0][0]
    y = tree.get_node(nid=node_id).data[0][1]

    if (((x - 1, y) in pass_point) == 0):
        simulation_avail_move[0] = 0
    if (((x + 1, y) in pass_point) == 0):
        simulation_avail_move[1] = 0
    if (((x, y - 1) in pass_point) == 0):
        simulation_avail_move[2] = 0
    if (((x, y + 1) in pass_point) == 0):
        simulation_avail_move[3] = 0

    return (simulation_avail_move)

def get_moni_move(node_id):  # 获得可模拟的操作

    simulation_avail_move = [1, 2, 3, 4]
    x = tree.get_node(nid=node_id).data[0][0]
    y = tree.get_node(nid=node_id).data[0][1]

    if (((x - 1, y) in pass_point) == 0):
        simulation_avail_move[0] = 0
    if (((x + 1, y) in pass_point) == 0):
        simulation_avail_move[1] = 0
    if (((x, y - 1) in pass_point) == 0):
        simulation_avail_move[2] = 0
    if (((x, y + 1) in pass_point) == 0):
        simulation_avail_move[3] = 0

    if (((x - 1, y) in coveraged) == 1):
        simulation_avail_move[0] = 0
    if (((x + 1, y) in coveraged) == 1):
        simulation_avail_move[1] = 0
    if (((x, y - 1) in coveraged) == 1):
        simulation_avail_move[2] = 0
    if (((x, y + 1) in coveraged) == 1):
        simulation_avail_move[3] = 0

    return (simulation_avail_move)

def p_cal(node_id):
    # todo reward和p的计算肯定有问题
    coveraged = []
    parent_id = tree.parent(nid=node_id).identifier
    while parent_id != 'root':
        coveraged.append(tree.get_node(nid=parent_id).data[0])
        parent_id = tree.parent(nid=parent_id).identifier

    coordinate = tree.get_node(nid=node_id).data[0]
    if coordinate in coveraged:
        p = 0
    else:
        p = 1

    return p

def is_turn(node_id):   # 判断机器人在这一步是否转向 , 计算转向损失值rk
    parent_move = tree.get_node(node_id).data[3]
    rk = 0

    parent_id = tree.parent(node_id).identifier
    grandpa_move = tree.get_node(parent_id).data[3]

    if parent_move != grandpa_move:  # 转向rk=1 后退rk=10
        rk = 0.2
        if ((parent_move+grandpa_move) ==7) or ((parent_move+grandpa_move) ==3):
            rk = 0.5
            # todo

    return rk

# def reward_cal(simulation_id,expand_id):  # reward = Σ[p/(tk+1)^2]
#
#     # todo reward计算过程错了
#     dt, k = 0.1, 0
#     reward = 0
#     node_id = simulation_id
#
#     while node_id != expand_id:
#         p = p_cal(node_id)
#         k = tree.depth(tree.get_node(nid=node_id))-tree.depth(tree.get_node(nid=expand_id))
#         tk = dt * k
#         rk = 0  #is_turn(node_id)
#         reward += (p-rk)/((tk+1)**2)
#         node_id = tree.parent(node_id).identifier
#
#     return reward

def reward_cal(node_id):  # reward = Σ[p/(tk+1)^2]

    dt, k = 0.1, 0
    reward = 0

    while node_id != 'root':   # 计算路径上的奖励值
        p = p_cal(node_id)
        k = tree.depth(tree.get_node(nid=node_id))
        tk = dt * k
        rk = 0 #is_turn(node_id)
        reward += (p-rk)/((tk+1)**2)
        node_id = tree.parent(node_id).identifier

        # print('reward:',reward)

    return reward


def UCT_cal(node_id):

    reward = tree.get_node(nid=node_id).data[1]
    visit = tree.get_node(nid=node_id).data[2]
    parent_id=tree.parent(nid=node_id).identifier
    parent_visit=tree.get_node(nid=parent_id).data[2]

    # todo 改变c值
    # c=1/(math.sqrt(2))
    c = 0.00001

    left = reward / visit
    right = c*math.sqrt(2*math.log(parent_visit)/visit)

    # print('q:',left,'ucb:',right)

    UCT =  left +right
    #todo
    data = tree.get_node(node_id).data
    tree.update_node(nid=node_id,data=[data[0],data[1],data[2],data[3],UCT]) #更新UCT值

    return UCT

def coverage_cal(node_id):   #计算有效的覆盖栅格数

    coverage = 0
    # if (tree.get_node(nid=node_id).is_leaf()):

    explore_trails=[(1,1)]

    while (tree.get_node(nid=node_id).is_root()==False):  # 判断当前节点是否是根节点
        trail=tree.get_node(nid=node_id).data[0]
        explore_trails.append(trail)
        node_id=tree.parent(nid=node_id).identifier

    l1=explore_trails
    for el in range(len(l1) - 1, -1, -1):
        if l1.count(l1[el]) > 1:
            l1.pop(el)

    coverage = len(l1)

    return (coverage)

def select(node_id):  # 输入当前节点，选择最优的子节点id

    children = tree.children(node_id)
    children_UCT = [0, 0, 0, 0]
    for i in range(len(children)):

        children_UCT[i] = UCT_cal(children[i].identifier)

    best_j = 0

    max_UCT = children_UCT[0]
    # todo 改进UCT选择过程
    for j in range(len(children)):
        if children_UCT[j] >= max_UCT:
            max_UCT = children_UCT[j]
            best_j = j

    return (children[best_j].identifier)



def backup(leaf_node_id, reward):  # 从叶子节点出发，回溯更新整棵树
    node_id = leaf_node_id
    parent_id = tree.parent(nid=node_id).identifier
    while (tree.get_node(nid=parent_id).is_root() == 0):  # 判断当前节点的父节点是否是根节点
        tree.get_node(nid=parent_id).data[1] += reward  # 更新reward
        tree.get_node(nid=parent_id).data[2] += 1  # 更新visit_times
        node_id = tree.parent(nid=node_id).identifier
        parent_id = tree.parent(nid=parent_id).identifier
    tree.get_node(nid='root').data[1] += reward
    tree.get_node(nid='root').data[2] += 1


'''
2.MCTS过程
'''
def MCTS_select(root_id):  # 输入根节点id,清空select_node_trails[],重新计算并输出根节点到当前扩展层的最优节点序列select_node_trails

    node_id = root_id
    root_node = tree.get_node(nid=root_id)

    select_coordinate_trails = [(1, 1)]  # 坐标序列
    select_node_trails = [root_node]  # 节点序列

    while (tree.get_node(nid=node_id).is_leaf() == 0):  # 当前节点是叶子节点,选择过程结束
        select_node_trail = tree.get_node(nid=select(node_id))
        select_node_trails.append(select_node_trail)

        select_coordinate_trail = tree.get_node(nid=select(node_id)).data[0]
        select_coordinate_trails.append(select_coordinate_trail)

        node_id = select_node_trail.identifier

    select_trails = [select_node_trails, select_coordinate_trails]
    return (select_trails)


def MCTS_expand(node_id):
    sub_id = tree_update(node_id, get_expand_move(node_id))
    return (sub_id)  # 返回子节点id
'''
模拟过程结束判定：
1.整个树的层数达到200或者仿真结束
模拟过程策略：
随机策略 vs 内螺旋策略
'''
def MCTS_simulation(node_id):  # 从扩展得到的节点出发向下模拟
    simulation_trails_id = []  # 仿真过程的节点id序列
    simulation_trails_coordinate = []  # 仿真过程的节点坐标序列

    depth = tree.depth(node=tree.get_node(node_id))
    while depth <= 500:  # 保证模拟过程后，树的总层数为200
        sub_id = default_policy(node_id)
        simulation_trails_id.append(sub_id)
        simulation_trails_coordinate.append(tree.get_node(nid=sub_id).data[0])
        node_id = sub_id
        depth = tree.depth(node=tree.get_node(node_id))
        mcts_coverage = coverage_cal(sub_id) / len(pass_point)
        # print('仿真覆盖率:', mcts_coverage)
        if mcts_coverage == 1.0:
            print('程序结束')
            explore_trails = [(1, 1)]

            node_id = sub_id
            while (tree.get_node(nid=node_id).is_root() == False):  # 判断当前节点是否是根节点
                trail = tree.get_node(nid=node_id).data[0]
                explore_trails.append(trail)
                node_id = tree.parent(nid=node_id).identifier
            print('仿真得到的机器人路径：', explore_trails)
            print('重复率：', len(explore_trails) / len(pass_point))
            sys.exit(0)



    # print('仿真节点数：',len(simulation_trails_coordinate))

    simulation_trails = [simulation_trails_id, simulation_trails_coordinate]
    return (simulation_trails)


# def MCTS_backup(simulation_id, expand_id):
#     reward = reward_cal(simulation_id,expand_id)
#     backup(expand_id, reward)
def MCTS_backup(simulation_id, expand_id):
    reward = reward_cal(simulation_id) - reward_cal(expand_id)
    backup(expand_id, reward)


def main(root_node_id, expect_coverage):  # 输入起始节点和期望覆盖率,输出最优路径

    node_id = root_node_id
    mcts_coverage = 0
    step = 1
    x,y=[],[]
    Expect_coverage = expect_coverage
    select_trails = []

    while (mcts_coverage < Expect_coverage):

        ex_avail_move = get_expand_move(node_id)
        while (ex_avail_move != [0, 0, 0, 0]):
            expand_node_id = MCTS_expand(node_id)
            ex_avail_move = get_expand_move(node_id)

            simulation_trails = MCTS_simulation(expand_node_id)[0]  # 模拟得到的节点队列
            if len(simulation_trails) == 0:
                backup(expand_node_id, 0)

            # todo 随机报错
            elif coverage_cal(simulation_trails[-1]) < 0.5:
                backup(expand_node_id ,0)
            else:
                simulation_leaf_node_id = simulation_trails[-1]  # 模拟过程的叶子节点

                MCTS_backup(simulation_leaf_node_id, expand_node_id)  # 从模拟得到的叶子节点开始回溯


                # 仿真覆盖率计算
                # mcts_coverage = coverage_cal(simulation_leaf_node_id) / len(pass_point)
                # print('仿真覆盖率:', mcts_coverage)
                # if mcts_coverage == 1.0:
                #     print('程序结束')
                #     explore_trails = [(1, 1)]
                #
                #     node_id = simulation_leaf_node_id
                #     while (tree.get_node(nid=node_id).is_root() == False):  # 判断当前节点是否是根节点
                #         trail = tree.get_node(nid=node_id).data[0]
                #         explore_trails.append(trail)
                #         node_id = tree.parent(nid=node_id).identifier
                #     print('仿真得到的机器人路径：',explore_trails)
                #     print('重复率：',len(explore_trails)/len(pass_point))
                #     sys.exit(0)

                '''
                仿真结束后去除simulation过程创造的节点,使得蒙特卡洛树只保留expand得到的节点
                '''
                expand_child_id = tree.children(nid=expand_node_id)[0].identifier
                tree.remove_subtree(nid=expand_child_id)

            # tree.show()
            # todo  print('树节点数量：',len(tree.all_nodes()))
        # todo print(tree.children('root'))

        z = MCTS_select(root_node_id)
        select_id = z[0][-1].identifier
        select_trails =z[1]

        # print(z[1])

        # tree.show()

        # print(tree.depth(tree.get_node(nid=select_id)))   # 测试用，用后即删
        # print(tree.parent(nid=select_id))

        mcts_coverage = coverage_cal(select_id) / len(pass_point)
        print('覆盖率:', mcts_coverage)

        # x.append(step)
        # y.append(mcts_coverage)
        # step += 1

        node_id = select_id  # 更新mcts过程的开始节点

    # todo 绘制单机器人实时覆盖率
    # plt.plot(x,y)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.title('单机器人实时覆盖率',loc='center')
    # plt.xlabel('机器人覆盖步长/（步）')
    # plt.ylabel('覆盖率/（无单位）')
    # plt.savefig("单机器人实时覆盖率.png")
    # plt.clf()
    # plt.cla()
    # plt.close('all')

    return (select_trails)

# 初始化
AVAILABLE_CHOICES = [1, 2, 3, 4]  # 定义可选操作集合--->映射到机器人四个可选动作 上下左右

pass_point = [(1, 1), (1, 2), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 1), (2, 2), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 1), (4, 2), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),  (5, 9), (5, 10), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10)]
obstacles = [(1,3),(2,3),(3,3),(4,3),(5,6),(5,7),(5,8)]
# pass_point =  [(1, 1), (1, 2), (1, 4), (1, 5), (2, 1), (2, 2) ,(2, 4), (2, 5), (3, 1), (3, 4), (3, 5), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
# obstacles = [(1,3),(2,3),(3,2),(3,3)]
coveraged = [(1,1)]
# pass_point= [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 2), (2, 3), (2, 6), (3, 1), (3, 2), (3, 3), (3, 5), (3, 6), (3, 7), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (5, 1), (5, 2), (5, 3), (5, 5), (5, 6), (5, 7), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]

# obstacles=[(8, 8), (5, 10), (2, 3), (8, 5), (6, 2), (5, 4), (10, 1), (10, 6), (10, 3), (6, 1), (2, 5), (8, 4), (10, 9), (5, 7), (4, 4), (4, 8), (3, 4), (10, 8), (2, 4), (7, 4)]
# pass_point=[(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 1), (2, 2), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 1), (3, 2), (3, 3), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (4, 7), (4, 9), (4, 10), (5, 1), (5, 2), (5, 3), (5, 5), (5, 6), (5, 8), (5, 9), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (7, 1), (7, 2), (7, 3), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (8, 1), (8, 2), (8, 3), (8, 6), (8, 7), (8, 9), (8, 10), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (10, 2), (10, 4), (10, 5), (10, 7), (10, 10)]



# 树结构定义
tree = Tree()
tree.create_node(tag='root', identifier='root', parent=None,
                 data=[(1, 1), 0, 1, 1, 1])  # data[0]-坐标 data[1]-reward data[2]-visit data[3]-parent_move data[4]-UCT

expect_coverage = 1.0
round = 1

start = time.perf_counter()
robot_path = main('root', expect_coverage)  # 机器人覆盖路径
end = time.perf_counter()
print('程序运行时间：', end - start)

repeat = len(robot_path) / (len(pass_point) * expect_coverage)  # 机器人重复覆盖率
length = len(robot_path)
print('机器人路径：', robot_path)
print('重复覆盖率：', repeat)
print('覆盖步长:', length)

tree.save2file(filename='tree')

