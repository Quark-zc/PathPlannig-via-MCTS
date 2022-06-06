import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors

# 单个机器人覆盖绘图

def draw_map(rows, cols, pass_point, trails, obstacles):
    field = np.ones((rows, cols))
    for i in range(len(pass_point)):
        field[pass_point[i][0] - 1, pass_point[i][1] - 1] = 1  # 注释：横纵坐标-1是因为障碍物和通行点的坐标定义是从下标1开始的,但field()函数下标从1开始
    for j in range(len(trails)):
        field[trails[j][0] - 1, trails[j][1] - 1] = 4
    for k in range(len(obstacles)):
        field[obstacles[k][0] - 1, obstacles[k][1] - 1] = 2
    # 显示当前机器人位置
    # field[trails[j][0] - 1, trails[j][1] - 1] = 6

    cmap = colors.ListedColormap(['none', 'white', 'black', 'red', 'yellow', 'magenta', 'green', 'cyan', 'blue'])

    plt.figure(figsize=(cols, rows))
    ax = plt.gca()

    sns.heatmap(field, cmap=cmap, vmin=0, vmax=8, linewidths=1.25, linecolor='black', ax=ax, cbar=False)

    # 设置图标题
    ax.set_ylabel('rows')
    ax.set_xlabel('cols')

    # 将列标签移动到图像上方
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # 设置图标的数字个数文字，放在plt.show下面能居中
    ax.set_xticks(np.arange(rows))
    ax.set_yticks(np.arange(cols))

    return field

def draw_path(path):
    # todo 改好绘图函数
    draw_map(rows, cols, pass_point, trails, obstacles)
    for i in range(len(path)-1):


        x = path[i][1]   # x,y需要反过来
        y = path[i][0]
        dx = path[i+1][1] - x
        dy = path[i+1][0] - y
        plt.arrow(x-0.5,y-0.5,dx,dy,width=0.1,head_width=0.1,head_length=0.1)
        # plt.arrow(x-0.5,y-0.5,dx,dy,width=0.01)



def main(rows, cols, pass_point, trails, obstacles):

    # plt.savefig('map_10x10.png')
    draw_path(trails)
    plt.show()

    # plt.arrow(3, 2, -1, -1, width=0.001, head_width=0.005, head_length=0.1)
    # plt.show()

# trails = []
# rows, cols = 10, 10
# obstacles=[(8, 8), (5, 10), (2, 3), (8, 5), (6, 2), (5, 4), (10, 1), (10, 6), (10, 3), (6, 1), (2, 5), (8, 4), (10, 9), (5, 7), (4, 4), (4, 8), (3, 4), (10, 8), (2, 4), (7, 4)]
# pass_point=[(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 1), (2, 2), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 1), (3, 2), (3, 3), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (4, 7), (4, 9), (4, 10), (5, 1), (5, 2), (5, 3), (5, 5), (5, 6), (5, 8), (5, 9), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (7, 1), (7, 2), (7, 3), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (8, 1), (8, 2), (8, 3), (8, 6), (8, 7), (8, 9), (8, 10), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (10, 2), (10, 4), (10, 5), (10, 7), (10, 10)]

# obstacles= [(5, 8), (8, 7), (6, 1), (10, 2), (7, 7), (6, 2), (6, 10), (9, 7), (1, 4), (3, 10), (6, 4), (10, 8), (8, 3), (6, 5), (4, 4), (3, 7), (4, 3), (2, 10), (2, 7), (5, 5), (4, 1), (2, 6), (6, 9), (1, 3), (10, 4)]
# pass_point= [(1, 1), (1, 2), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 8), (2, 9), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 8), (3, 9), (4, 2), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (5, 1), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 9), (5, 10), (6, 3), (6, 6), (6, 7), (6, 8), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 8), (7, 9), (7, 10), (8, 1), (8, 2), (8, 4), (8, 5), (8, 6), (8, 8), (8, 9), (8, 10), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 8), (9, 9), (9, 10), (10, 1), (10, 3), (10, 5), (10, 6), (10, 7), (10, 9), (10, 10)]

# obstacles= [(4, 5), (2, 7), (7, 3), (4, 8), (1, 1), (3, 4), (10, 7), (8, 8), (10, 1), (1, 8), (3, 1), (7, 1), (10, 3), (9, 6), (3, 7)]
# pass_point= [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 9), (1, 10), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 8), (2, 9), (2, 10), (3, 2), (3, 3), (3, 5), (3, 6), (3, 8), (3, 9), (3, 10), (4, 1), (4, 2), (4, 3), (4, 4), (4, 6), (4, 7), (4, 9), (4, 10), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (7, 2), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 9), (8, 10), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 7), (9, 8), (9, 9), (9, 10), (10, 2), (10, 4), (10, 5), (10, 6), (10, 8), (10, 9), (10, 10)]

# obstacles=[(8, 8), (5, 10), (2, 3), (8, 5), (6, 2), (5, 4), (10, 1), (10, 6), (10, 3), (6, 1), (2, 5), (8, 4), (10, 9), (5, 7), (4, 4), (4, 8), (3, 4), (10, 8), (2, 4), (7, 4)]
# pass_point=[(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 1), (2, 2), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 1), (3, 2), (3, 3), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (4, 7), (4, 9), (4, 10), (5, 1), (5, 2), (5, 3), (5, 5), (5, 6), (5, 8), (5, 9), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (7, 1), (7, 2), (7, 3), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (8, 1), (8, 2), (8, 3), (8, 6), (8, 7), (8, 9), (8, 10), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (10, 2), (10, 4), (10, 5), (10, 7), (10, 10)]
# rows,cols=10,10
# trails = pass_point


# pass_point =  [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
# obstacles = []
# rows, cols =5, 5
# obstacles=[(8, 8), (5, 10), (2, 3), (8, 5), (6, 2), (5, 4), (10, 1), (10, 6), (10, 3), (6, 1), (2, 5), (8, 4), (10, 9), (5, 7), (4, 4), (4, 8), (3, 4), (10, 8), (2, 4), (7, 4)]
# pass_point=[(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 1), (2, 2), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 1), (3, 2), (3, 3), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (4, 7), (4, 9), (4, 10), (5, 1), (5, 2), (5, 3), (5, 5), (5, 6), (5, 8), (5, 9), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (7, 1), (7, 2), (7, 3), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (8, 1), (8, 2), (8, 3), (8, 6), (8, 7), (8, 9), (8, 10), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (10, 2), (10, 4), (10, 5), (10, 7), (10, 10)]
rows,cols=10,10
# # pass_point = [(1, 1), (1, 2), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 1), (2, 2), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 1), (4, 2), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10)]
# # obstacles = [(1,3),(2,3),(3,3),(4,3)]
pass_point = [(1, 1), (1, 2), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 1), (2, 2), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 1), (4, 2), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),  (5, 9), (5, 10), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10)]
obstacles = [(1,3), (2,3), (3,3), (4,3), (5,6), (5,7), (5,8)]
# # trails= [(3, 6), (3, 7), (3, 8), (4, 8), (4, 7), (4, 6), (4, 5), (3, 5), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (3, 9), (4, 9), (5, 9), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10), (10, 9), (10, 8), (10, 7), (10, 6), (10, 5), (10, 4), (10, 3), (10, 2), (10, 1), (9, 1), (8, 1), (7, 1), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (7, 8), (8, 8), (9, 8), (10, 8), (9, 8), (8, 8), (7, 8), (7, 7), (8, 7), (9, 7), (9, 8), (9, 9), (8, 9), (7, 9), (6, 9), (6, 8), (6, 7), (6, 6), (7, 6), (8, 6), (9, 6), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (9, 10), (8, 10), (7, 10), (6, 10), (5, 10), (4, 10), (3, 10), (2, 10), (1, 10), (1, 9), (1, 8), (1, 7), (1, 6), (1, 5), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (9, 4), (10, 4), (9, 4), (8, 4), (7, 4), (6, 4), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3), (10, 4), (10, 5), (9, 5), (8, 5), (7, 5), (6, 5), (5, 5), (5, 4), (5, 3), (5, 2), (4, 2), (3, 2), (2, 2), (1, 2), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (10, 2), (9, 2), (8, 2), (7, 2), (6, 2), (5, 2), (4, 2), (3, 2), (2, 2), (1, 2)]
#
# # trails = [(1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (1, 2), (1, 3), (1, 2), (2, 2), (2, 1), (3, 1), (3, 2), (4, 2), (5, 2), (4, 2), (4, 1), (5, 1), (5, 2), (5, 3), (5, 4), (4, 4), (5, 4), (5, 5), (5, 4), (5, 5), (5, 4), (5, 3), (4, 3), (4, 4), (3, 4), (3, 3), (2, 3), (2, 4), (1, 4), (2, 4), (2, 5), (1, 5), (2, 5), (3, 5), (4, 5)]
#
# # trails = [(1, 1), (2, 1), (3, 1), (2, 1), (2, 2), (3, 2), (4, 2), (4, 1), (3, 1), (2, 1), (2, 2), (3, 2), (3, 1), (3, 2), (3, 1), (2, 1), (3, 1), (4, 1), (4, 2), (4, 1), (5, 1), (5, 2), (4, 2), (4, 1), (3, 1), (2, 1), (3, 1), (2, 1), (2, 2), (2, 1), (2, 2), (3, 2), (4, 2), (3, 2), (3, 1), (3, 2), (4, 2), (5, 2), (5, 1), (4, 1), (3, 1), (2, 1), (3, 1), (2, 1), (1, 1), (1, 2), (1, 1), (2, 1), (3, 1), (4, 1), (4, 2), (5, 2), (4, 2), (5, 2), (4, 2), (4, 3), (4, 2), (3, 2), (4, 2), (3, 2), (2, 2), (3, 2), (2, 2), (3, 2), (4, 2), (3, 2), (3, 3), (4, 3), (3, 3), (3, 2), (2, 2), (3, 2), (3, 3), (3, 2), (2, 2), (3, 2), (4, 2), (5, 2), (5, 1), (4, 1), (5, 1), (5, 2), (4, 2), (5, 2), (5, 3), (6, 3), (6, 4), (6, 5), (7, 5), (6, 5), (7, 5), (6, 5), (6, 6), (7, 6), (6, 6), (5, 6), (4, 6), (4, 5), (3, 5), (4, 5), (4, 6), (5, 6), (6, 6), (6, 7), (6, 6), (6, 5), (7, 5), (6, 5), (6, 4), (6, 3), (6, 4), (6, 3), (7, 3), (7, 2), (7, 1), (8, 1), (7, 1), (8, 1), (8, 2), (9, 2), (10, 2), (9, 2), (8, 2), (9, 2), (9, 1), (9, 2), (9, 1), (8, 1), (9, 1), (8, 1), (7, 1), (7, 2), (8, 2), (8, 1), (8, 2), (7, 2), (8, 2), (7, 2), (8, 2), (8, 3), (8, 2), (7, 2), (8, 2), (9, 2), (9, 1), (9, 2), (10, 2), (9, 2), (8, 2), (7, 2), (8, 2), (8, 1), (8, 2), (8, 3), (9, 3), (9, 2), (8, 2), (9, 2), (9, 1), (8, 1), (7, 1), (8, 1), (8, 2), (8, 3), (9, 3), (8, 3), (9, 3), (9, 2), (9, 1), (8, 1), (8, 2), (9, 2), (10, 2), (9, 2), (9, 1), (8, 1), (9, 1), (9, 2), (9, 1), (8, 1), (7, 1), (7, 2), (7, 1), (8, 1), (8, 2), (7, 2), (7, 3), (8, 3), (9, 3), (8, 3), (8, 2), (8, 1), (9, 1), (9, 2), (8, 2), (9, 2), (10, 2), (9, 2), (8, 2), (8, 1), (9, 1), (8, 1), (7, 1), (7, 2), (8, 2), (8, 3), (8, 2), (9, 2), (8, 2), (7, 2), (7, 1), (7, 2), (7, 3), (6, 3), (6, 4), (6, 3), (6, 4), (6, 3), (5, 3), (6, 3), (7, 3), (8, 3), (9, 3), (9, 4), (10, 4), (9, 4), (10, 4), (10, 5), (9, 5), (9, 6), (9, 7), (8, 7), (8, 6), (9, 6), (8, 6), (9, 6), (9, 7), (10, 7), (9, 7), (8, 7), (9, 7), (9, 8), (9, 7), (8, 7), (7, 7), (7, 6), (7, 5), (7, 6), (8, 6), (9, 6), (9, 5), (9, 6), (9, 7), (9, 6), (9, 5), (10, 5), (10, 4), (10, 5), (10, 4), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (8, 9), (8, 10), (8, 9), (7, 9), (6, 9), (5, 9), (6, 9), (6, 10), (6, 9), (5, 9), (6, 9), (6, 10), (7, 10), (7, 9), (7, 8), (7, 9), (7, 8), (7, 7), (8, 7), (8, 6), (9, 6), (8, 6), (8, 7), (8, 6), (8, 7), (8, 6), (7, 6), (8, 6), (8, 7), (8, 6), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (8, 10), (8, 9), (8, 10), (7, 10), (8, 10), (9, 10), (9, 9), (9, 8), (9, 7), (8, 7), (8, 6), (9, 6), (8, 6), (8, 7), (9, 7), (9, 8), (9, 7), (9, 6), (8, 6), (7, 6), (7, 5), (7, 6), (8, 6), (9, 6), (9, 5), (10, 5), (10, 4), (9, 4), (9, 5), (10, 5), (10, 4), (10, 5), (10, 4), (10, 5), (10, 4), (9, 4), (9, 5), (10, 5), (9, 5), (9, 4), (9, 5), (9, 6), (9, 5), (10, 5), (10, 4), (9, 4), (9, 5), (9, 4), (9, 3), (9, 4), (9, 5), (9, 4), (9, 3), (9, 2), (9, 3), (9, 4), (9, 5), (9, 4), (9, 3), (8, 3), (9, 3), (9, 4), (10, 4), (10, 5), (10, 4), (9, 4), (10, 4), (9, 4), (9, 3), (8, 3), (9, 3), (9, 2), (9, 3), (9, 2), (8, 2), (8, 1), (9, 1), (9, 2), (10, 2), (9, 2), (9, 3), (9, 2), (9, 3), (9, 2), (9, 1), (9, 2), (10, 2), (9, 2), (8, 2), (9, 2), (8, 2), (8, 1), (8, 2), (7, 2), (7, 3), (6, 3), (6, 4), (6, 3), (7, 3), (7, 2), (7, 3), (6, 3), (5, 3), (6, 3), (6, 4), (6, 3), (7, 3), (6, 3), (7, 3), (8, 3), (9, 3), (9, 4), (9, 3), (8, 3), (8, 2), (8, 3), (8, 2), (7, 2), (8, 2), (8, 1), (7, 1), (7, 2), (8, 2), (7, 2), (7, 1), (7, 2), (7, 3), (7, 2), (7, 3), (6, 3), (5, 3), (5, 2), (4, 2), (4, 1), (3, 1), (3, 2), (3, 3), (3, 2), (4, 2), (5, 2), (5, 3), (4, 3), (4, 2), (4, 3), (4, 2), (4, 3), (5, 3), (6, 3), (6, 4), (6, 3), (6, 4), (6, 3), (6, 4), (6, 5), (6, 4), (6, 5), (6, 6), (6, 5), (6, 6), (7, 6), (7, 7), (6, 7), (6, 6), (7, 6), (7, 7), (8, 7), (7, 7), (6, 7), (6, 8), (5, 8), (5, 9), (5, 8), (6, 8), (7, 8), (7, 9), (7, 10), (7, 9), (7, 8), (7, 7), (8, 7), (9, 7), (9, 8), (9, 7), (9, 8), (9, 9), (8, 9), (7, 9), (6, 9), (6, 8), (5, 8), (6, 8), (6, 7), (7, 7), (8, 7), (8, 6), (7, 6), (6, 6), (7, 6), (8, 6), (8, 7), (7, 7), (7, 8), (6, 8), (5, 8), (6, 8), (6, 7), (6, 6), (6, 5), (5, 5), (6, 5), (5, 5), (6, 5), (7, 5), (7, 6), (6, 6), (6, 7), (6, 8), (7, 8), (7, 9), (6, 9), (7, 9), (8, 9), (9, 9), (9, 10), (9, 9), (9, 10), (8, 10), (7, 10), (8, 10), (9, 10), (9, 9), (8, 9), (8, 10), (8, 9), (7, 9), (7, 10), (6, 10), (7, 10), (8, 10), (8, 9), (9, 9), (9, 10), (9, 9), (8, 9), (8, 10), (7, 10), (7, 9), (7, 10), (8, 10), (9, 10), (10, 10), (9, 10), (8, 10), (7, 10), (6, 10), (6, 9), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10), (9, 10), (9, 9), (9, 8), (9, 9), (9, 10), (9, 9), (9, 8), (9, 9), (9, 10), (8, 10), (9, 10), (9, 9), (8, 9), (8, 10), (7, 10), (6, 10), (6, 9), (6, 8), (6, 9), (6, 10), (7, 10), (8, 10), (7, 10), (7, 9), (7, 8), (7, 7), (6, 7), (6, 8), (6, 9), (5, 9), (5, 8), (5, 9), (4, 9), (5, 9), (6, 9), (6, 10), (7, 10), (6, 10), (6, 9), (5, 9), (6, 9), (5, 9), (5, 8), (6, 8), (7, 8), (7, 7), (7, 8), (6, 8), (6, 9), (6, 10), (6, 9), (5, 9), (4, 9), (5, 9), (4, 9), (3, 9), (2, 9), (3, 9), (2, 9), (1, 9), (1, 10), (2, 10), (1, 10), (1, 9), (1, 10), (1, 9), (2, 9), (2, 8), (3, 8), (2, 8), (2, 7), (2, 8), (1, 8), (1, 7), (2, 7), (1, 7), (2, 7), (1, 7), (2, 7), (1, 7), (1, 8), (1, 9), (2, 9), (2, 8), (1, 8), (2, 8), (2, 7), (2, 8), (2, 9), (1, 9), (2, 9), (2, 10), (3, 10), (4, 10), (4, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (7, 8), (7, 7), (8, 7), (7, 7), (6, 7), (6, 6), (6, 5), (7, 5), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (5, 9), (4, 9), (4, 10), (4, 9), (3, 9), (2, 9), (2, 10), (3, 10), (4, 10), (3, 10), (3, 9), (4, 9), (4, 10), (3, 10), (3, 9), (3, 10), (3, 9), (3, 10), (2, 10), (2, 9), (2, 8), (2, 7), (2, 8), (1, 8), (1, 7), (2, 7), (1, 7), (1, 8), (1, 7), (2, 7), (2, 6), (3, 6), (2, 6), (3, 6), (3, 7), (4, 7), (3, 7), (3, 6), (4, 6), (4, 5), (4, 6), (4, 7), (4, 6), (4, 5), (5, 5), (4, 5), (3, 5), (3, 6), (4, 6), (3, 6), (3, 5), (4, 5), (5, 5), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 8), (7, 8), (7, 9), (7, 8), (7, 9), (8, 9), (9, 9), (9, 8), (9, 9), (9, 8), (9, 7), (8, 7), (7, 7), (7, 8), (7, 7), (7, 8), (7, 9), (7, 8), (6, 8), (6, 7), (6, 8), (6, 7), (6, 6), (7, 6), (6, 6), (5, 6), (4, 6), (3, 6), (2, 6), (1, 6), (1, 7), (2, 7), (2, 6), (3, 6), (2, 6), (2, 7), (1, 7), (1, 8), (1, 7), (1, 6), (1, 5), (1, 6), (1, 5), (1, 4)]
# obstacles = []
trails = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10), (10, 9), (10, 8), (10, 7), (10, 6), (10, 5), (10, 4), (10, 3), (10, 2), (10, 1), (9, 1), (8, 1), (7, 1), (6, 1), (5, 1), (4, 1), (3, 1), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9), (9, 8), (9, 7), (9, 6), (9, 5), (9, 4), (9, 3), (9, 2), (8, 2), (7, 2), (6, 2), (5, 2), (4, 2), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (8, 7), (8, 6), (8, 5), (8, 4), (8, 3), (7, 3), (6, 3), (5, 3), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (5, 7), (6, 7), (7, 7), (7, 6), (7, 5), (7, 4), (6, 4), (5, 4), (5, 5), (5, 6), (6, 6),(6, 5)]

# trails = [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (9, 1), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (7, 8), (7, 7), (7, 6), (7, 5), (8, 5), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (8, 9), (7, 9), (6, 9), (6, 8), (6, 7), (6, 6), (6, 5), (6, 4), (5, 4), (4, 4), (3, 4), (2, 4), (1, 4), (2, 4), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (4, 8), (4, 7), (4, 6), (4, 5), (3, 5), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (3, 9), (4, 9), (5, 9), (5, 8), (5, 7), (5, 6), (5, 5), (5, 4), (5, 3), (6, 3), (7, 3), (8, 3), (9, 3), (9, 4), (8, 4), (7, 4), (6, 4), (5, 4), (4, 4), (3, 4), (2, 4), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10), (10, 9), (10, 8), (10, 7), (10, 6), (10, 5), (10, 4), (10, 3), (10, 2), (9, 2), (8, 2), (7, 2), (6, 2), (5, 2), (4, 2), (3, 2), (2, 2), (1, 2),(1, 1)]


# trails=[(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (10, 2), (10, 3), (9, 3), (8, 3), (7, 3), (6, 3), (5, 3), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3), (10, 2), (9, 2), (8, 2), (7, 2), (6, 2), (5, 2), (4, 2), (3, 2), (2, 2), (1, 2)]
# pass_point = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2), (5, 3), (6, 1), (6, 2), (6, 3), (7, 1), (7, 2), (7, 3), (8, 1), (8, 2), (8, 3), (9, 1), (9, 2), (9, 3), (10, 1), (10, 2), (10, 3)]
# obstacles = [(1,3),(2,3),(3,3),(4,3)]  # 机器人1

# pass_point = [(1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (5, 4), (5, 5), (5, 9), (5, 10)]
# obstacles = [(5, 6), (5, 7), (5, 8)]  # 机器人2
# trails = [ (4, 6), (4, 7), (4, 8), (3, 8), (3, 7), (3, 6), (3, 5), (4, 5), (5, 5), (5, 4), (4, 4), (3, 4), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (3, 9), (4, 9), (5, 9), (5, 10), (4, 10), (3, 10), (2, 10), (1, 10), (1, 9), (1, 8), (1, 7), (1, 6), (1, 5),(1,4)]

# pass_point = [(6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10)]
# obstacles = []
# trails = [(8,8),(8, 7), (8, 6), (7, 6), (7, 7), (7, 8), (7, 9), (8, 9), (9, 9), (9, 8), (9, 7), (9, 6), (9, 5), (8, 5), (7, 5), (6, 5), (6, 4), (7, 4), (8, 4), (9, 4), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (9, 10), (8, 10), (7, 10), (6, 10), (6, 9), (6, 8), (6, 7), (6, 6), (6, 5)]  # 机器人3

# pass_point = [(1, 1), (1, 2), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 1), (2, 2), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 1), (4, 2), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),  (5, 9), (5, 10), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10)]
# obstacles = [(1,3),(2,3),(3,3),(4,3),(5,6),(5,7),(5,8)]
# obstacles = []
# trails  = [(1, 1), (1, 2), (1, 1), (2, 1), (1, 1), (2, 1), (2, 2), (1, 2), (1, 3), (1, 4), (2, 4), (1, 4), (1, 5), (1, 4), (1, 5),
#  (2, 5), (3, 5), (2, 5), (3, 5), (4, 5), (4, 4), (4, 5), (4, 4), (4, 3), (5, 3), (5, 4), (4, 4), (4, 3), (5, 3), (4, 3),
#  (4, 2), (4, 3), (5, 3), (5, 2), (5, 1), (4, 1), (4, 2), (3, 2), (4, 2), (3, 2), (3, 1), (3, 2), (3, 3), (2, 3), (2, 4),
#  (2, 3), (3, 3), (3, 4), (4, 4), (5, 4), (5, 5)]
# trails = [(1, 1), (2, 1), (3, 1), (2, 1), (3, 1), (2, 1), (1, 1), (2, 1), (3, 1), (4, 1), (3, 1), (4, 1), (5, 1), (5, 2), (4, 2), (4, 1), (4, 2), (5, 2), (5, 3), (6, 3), (7, 3), (8, 3), (7, 3), (7, 2), (6, 2), (6, 3), (6, 4), (7, 4), (7, 5), (7, 4), (7, 5), (7, 4), (8, 4), (9, 4), (10, 4), (10, 3), (10, 2), (9, 2), (10, 2), (10, 1), (10, 2), (9, 2), (9, 1), (8, 1), (8, 2), (9, 2), (8, 2), (7, 2), (6, 2), (6, 1), (5, 1), (5, 2), (6, 2), (6, 3), (5, 3), (5, 2), (6, 2), (5, 2), (4, 2), (5, 2), (6, 2), (5, 2), (6, 2), (7, 2), (7, 3), (7, 4), (8, 4), (9, 4), (8, 4), (7, 4), (7, 5), (8, 5), (8, 6), (8, 5), (8, 6), (9, 6), (9, 5), (10, 5), (9, 5), (10, 5), (9, 5), (9, 6), (9, 7), (10, 7), (9, 7), (10, 7), (9, 7), (9, 8), (9, 7), (9, 8), (10, 8), (9, 8), (9, 9), (9, 8), (9, 9), (9, 8), (9, 9), (8, 9), (9, 9), (9, 10), (8, 10), (7, 10), (8, 10), (7, 10), (6, 10), (7, 10), (8, 10), (7, 10), (7, 9), (8, 9), (8, 10), (8, 9), (8, 10), (8, 9), (9, 9), (10, 9), (9, 9), (8, 9), (8, 10), (7, 10), (7, 9), (7, 8), (7, 9), (6, 9), (5, 9), (4, 9), (4, 8), (3, 8), (3, 7), (2, 7), (2, 8), (2, 9), (3, 9), (2, 9), (2, 8), (1, 8), (2, 8), (3, 8), (3, 7), (2, 7), (1, 7), (1, 6), (2, 6), (3, 6), (2, 6), (1, 6), (1, 5), (2, 5), (1, 5), (1, 4), (1, 5), (1, 6), (1, 5), (1, 6), (2, 6), (2, 5), (2, 4), (1, 4), (1, 5), (1, 6), (2, 6), (3, 6), (4, 6), (3, 6), (3, 7), (2, 7), (2, 6), (1, 6), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (3, 6), (2, 6), (2, 5), (1, 5), (1, 4), (1, 5), (2, 5), (2, 4), (2, 5), (1, 5), (1, 4), (2, 4), (3, 4), (3, 5), (2, 5), (3, 5), (3, 4), (2, 4), (2, 5), (2, 6), (1, 6), (2, 6), (2, 7), (3, 7), (4, 7), (4, 6), (3, 6), (4, 6), (3, 6), (4, 6), (3, 6), (4, 6), (4, 5), (3, 5), (2, 5), (1, 5), (1, 6), (1, 5), (1, 4), (2, 4), (1, 4), (1, 5), (2, 5), (3, 5), (2, 5), (2, 4), (2, 5), (2, 6), (2, 7), (3, 7), (4, 7), (4, 8), (4, 7), (3, 7), (3, 6), (4, 6), (4, 5), (4, 4), (4, 5), (4, 4), (3, 4), (3, 5), (4, 5), (3, 5), (4, 5), (5, 5), (4, 5), (3, 5), (3, 6), (3, 7), (3, 6), (3, 5), (3, 4), (3, 5), (2, 5), (3, 5), (4, 5), (5, 5), (4, 5), (5, 5), (5, 4), (6, 4), (6, 5), (6, 4), (6, 3), (6, 2), (6, 3), (6, 2), (6, 3), (7, 3), (7, 4), (7, 3), (7, 4), (8, 4), (8, 3), (8, 2), (9, 2), (9, 3), (9, 2), (9, 3), (9, 2), (9, 3), (8, 3), (8, 4), (8, 5), (9, 5), (10, 5), (9, 5), (9, 6), (9, 7), (10, 7), (10, 8), (10, 7), (9, 7), (10, 7), (10, 6)]
# trails = [(1,1),(2,1),(3,1),(4,1),(5,1),(5,2),(4,2),(3,2),(2,2),(1,2)]   # 测试绘图效果

# trails = [(1, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (9, 4), (10, 4), (9, 4), (8, 4), (7, 4), (6, 4), (5, 4), (4, 4), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (4, 8), (4, 7), (4, 6), (3, 6), (2, 6), (2, 7), (2, 8), (2, 9), (3, 9), (4, 9), (5, 9), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10), (10, 9), (10, 8), (10, 7), (10, 6), (10, 5), (10, 4), (10, 3), (10, 2), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 9), (10, 8), (10, 7), (10, 6), (10, 5), (10, 4), (10, 3), (10, 2), (9, 2), (8, 2), (7, 2), (6, 2), (5, 2), (4, 2), (3, 2), (2, 2), (1, 2), (1, 1), (1, 2), (1, 1), (1, 2), (2, 2), (3, 2), (4, 2), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (8, 2), (8, 3), (8, 4), (7, 4), (7, 3), (7, 2), (8, 2), (9, 2), (9, 3), (9, 4), (9, 5), (8, 5), (7, 5), (6, 5), (5, 5), (5, 4), (5, 3), (5, 2), (5, 1), (4, 1), (3, 1)]
# trails = [(4, 10), (4, 10), (3, 10), (2, 10), (1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10), (10, 9), (10, 8), (10, 7), (10, 6), (10, 5), (10, 4), (10, 3), (10, 2), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 9), (10, 8), (10, 7), (10, 6), (10, 5), (10, 4), (10, 3), (10, 2), (10, 1), (9, 1), (8, 1), (7, 1), (6, 1), (5, 1), (4, 1), (3, 1), (2, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 9), (10, 8), (10, 7), (10, 6), (10, 5), (10, 4), (10, 3), (10, 2), (10, 1), (9, 1), (8, 1), (7, 1), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (5, 10), (4, 10), (3, 10), (2, 10), (1, 10), (1, 9), (1, 8), (1, 7), (1, 6), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 4), (4, 4), (3, 4)]
# trails= [(3, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10), (10, 9), (10, 8), (10, 7), (10, 6), (10, 5), (10, 4), (10, 3), (10, 2), (10, 1), (9, 1), (8, 1), (7, 1), (6, 1), (5, 1), (4, 1), (3, 1), (2, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (9, 1), (8, 1), (7, 1), (6, 1), (5, 1), (4, 1), (3, 1), (2, 1), (1, 1), (1, 2), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (9, 1), (8, 1), (7, 1), (6, 1), (5, 1), (4, 1), (3, 1), (2, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (9, 9), (9, 8), (9, 7), (9, 6), (8, 6), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (8, 10), (9, 10), (10, 10), (10, 9), (10, 8), (10, 7), (10, 6), (10, 5), (10, 4), (10, 3), (10, 2), (10, 1), (9, 1), (8, 1), (7, 1), (6, 1), (6, 2)]


main(rows, cols, pass_point, trails, obstacles)