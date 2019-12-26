import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.interpolate as spi 
from scipy.linalg import solve

# 票房数据单位 万/周
weekZL_x = np.arange(12)
weekZL_y = np.array([99681.0, 217126.0, 140241.0, 56448.0, 26080.0, 17194.0, 5564.0,
                    2289.0, 1197.0, 619.0, 1029.0, 210.0])
assert weekZL_x.size == weekZL_y.size, "the size of ZL data is wrong"
weekLL_x = np.arange(14)
weekLL_y = np.array([0,202074.0, 177540.0, 56644.0, 17872.0, 6617.0, 2514.0, 1100.0,
                    462.0, 289.0, 127.0, 150.0, 41.0, 47.0]) 
assert weekLL_x.size == weekLL_y.size, "the size of LL data is wrong"

# 三次样条插值part
Dimension = weekLL_x.size -2 
h = 1 
Diag = 2*np.ones(Dimension)
ld = 0.5*np.ones(Dimension-1)
u3 = 0.5*np.ones(Dimension-1)
# 通过差商求出自然边界条件下的d的各项值
# 这一部分和上面那一部分讲道理可以改写成函数的。
d = np.zeros(Dimension)
for i in range(Dimension):
    d[i] = 3*(weekLL_y[i+2]-2*weekLL_y[i+1]+weekLL_y[i])
    pass
# print(d)


# 追赶法函数
def Chasing(a,b,c,d,show=True):
    # 初始化系数矩阵
    l = np.zeros(a.size)
    u = np.zeros(b.size)
    y = np.ones(d.size)
    M = np.zeros(y.size)
    u[0] = b[0]
    y[0] = d[0]
    # 求解系数矩阵
    for i in range(l.size):
        l[i] = a[i]/u[i]
        u[i+1] = b[i+1]-l[i]*c[i]
    # 求解ly=d
    for i in range(1,y.size):
        y[i] = d[i]-y[i-1]*l[i-1]
    # 求解UM=y
    M[M.size-1] = y[y.size-1]/u[y.size-1]
    for i in range(M.size-2,-1,-1):
        M[i] = (y[i]-c[i]*M[i+1])/u[i]
    # 控制参数显示
    if show==True:
        ShowAns(l,u,c,M)
    return M

# 结果展示函数
def ShowAns(l,u,c,M):
    print('*****************l**********')
    print(l)
    print('*****************u**********')
    print(u)
    print('*****************c**********')
    print(c)
    print('*****************M**********')
    print(M)
    pass

# 然后建立样条分段函数
# 并绘制图像
# 求出M
fake_M = Chasing(u3,Diag,ld,d,False)
M = np.zeros(fake_M.size+2)
M[1:M.size-1] = fake_M

# 建立三次样条插值函数（分段）
def MSpline3(data_x,data_y,M,h,x):
    assert x >= 0, "the x value is undefind(wrong)"
    index = x+1  
    index = int(index)
    if index==14: index=13
    V1 = pow(data_x[index]-x,3)*M[index-1]/6 
    V2 = pow(x-data_x[index-1],3)*M[index]/6
    V3 = (data_y[index-1]-M[index-1]/6)*(data_x[index]-x)
    V4 = (data_y[index]-M[index]/6)*(x-data_x[index-1])
    return V1+V2+V3+V4

# 绘制三次样条插值后的图像
x_plot = np.linspace(0,13,10000)
y_plot = np.array([MSpline3(weekLL_x,weekLL_y,M,h,t) for t in x_plot])
LineLens = 0
# Part2 不是数值积分，是局部线性化，数值积分是求面积的，憨憨
# for i in range(len(x_plot)-1):
#     temp= np.square(x_plot[i+1]-x_plot[i])+np.square(y_plot[i+1]-y_plot[i])
#     V = np.sqrt(temp)
#     LineLens = LineLens + V
# print('*************************')
# print(LineLens)
# plt.axis('scaled')
# plt.xlim((0,54))
# plt.ylim((-15,0))

plt.plot(x_plot,y_plot)
plt.scatter(weekLL_x,weekLL_y)
plt.show()


def my2ChIntegral(func,x_plot,threshold=500000,k=3):
    Ans = 0
    h = x_plot[1] - x_plot[0]
    for i in range(x_plot.size-1):
        Value1 = func(x_plot[i],para,k) + func(x_plot[i+1],para,k)
        temp = (x_plot[i]+x_plot[i+1])/2
        Value2 = func(temp,para,k)
        Ans = Ans + (Value1 + 4*Value2)*h/6  
    print(Ans)
    #     先算出目前的总和，然后用while控制变上限积分
    index = x_plot.size - 1 
    x_1 = x_plot[index]
    x_2 = x_plot[index] + h
    while(Ans<threshold):
        Value1 = func(x_1,para,k) + func(x_2,para,k)
        temp = (x_1+x_2)/2
        Value2 = func(temp,para,k)
        Ans = Ans + (Value1 + 4*Value2)*h/6
        x_1 = x_2
        x_2 = x_2 + h
        pass
    return x_2

def my3ChIntegral(func,x_plot,threshold=500000):
    Ans = 0 
    h = x_plot[1]-x_plot[0]
    print(x_plot.size-1)
    print(x_plot[x_plot.size-1])
    for i in range(x_plot.size-1):
        Value1 = func(weekLL_x,weekLL_y,M,h,x_plot[i]) + func(weekLL_x,weekLL_y,M,h,x_plot[i+1])
        temp = (x_plot[i]+x_plot[i+1])/2
        Value2 = func(weekLL_x,weekLL_y,M,h,temp)
        Ans = Ans + (Value1 + 4*Value2)*h/6  
    print(Ans)
    pass
Ans = my3ChIntegral(MSpline3,x_plot)