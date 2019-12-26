import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import scipy.interpolate as spi 

# 读入图像数据
x = np.arange(0,54,2)
y = np.array([0,4.01,6.96,7.96,7.97,8.02,9.05,10.13,11.18,12.26,13.28,12.61,10.22,7.90,7.95,8.86,10.80,10.93,
             11.23,11.30,10.94,10.10,9.54,8.30,7.30,2.50,0.20])
y = -y 
# 判断x，y的维度是否相等，如果不等就报出指定的data size错误
assert x.size == y.size, "the origin data is wrong. having different size" 

#  in this question, we know that h == 2
# 矩阵维度计算
Dimension = x.size-2 
# 由于本题的特殊性，dig=2，lamda=0.5=u,在此处就不需要考虑和原数字的index对位的问题了
# 后面还是需要考虑的
h = 2
Diag = 2*np.ones(Dimension)
ld = 0.5*np.ones(Dimension-1)
u3 = 0.5*np.ones(Dimension-1)
# 通过差商求出自然边界条件下的d的各项值
d = np.zeros(Dimension)
for i in range(Dimension):
    d[i] = 0.75*(y[i+2]-2*y[i+1]+y[i])
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
# 求出M，再添加自然边界条件（前后加0）
fake_M = Chasing(u3,Diag,ld,d,False)
M = np.zeros(fake_M.size+2)
M[1:M.size-1] = fake_M
# print(M)

# 建立三次样条插值函数（分段）
def MSpline3(data_x,data_y,M,h,x):
    assert x >= 0, "the x value is undefind(wrong)"
    # 以免输入错误的数据
    index = x//2 + 1 
    # 对应x的value和矩阵中次序的关系
    # 限制下标的范围（在边界最后一个值会有一个溢出，控制一下）
    index = int(index)
    if index==27: index=26
    # 根据M值建立三次样条插值的方程（带入方程表达式）
    V1 = pow(data_x[index]-x,3)*M[index-1]/12 
    V2 = pow(x-data_x[index-1],3)*M[index]/12
    V3 = (data_y[index-1]-4*M[index-1]/6)*(data_x[index]-x)/2
    V4 = (data_y[index]-4*M[index]/6)*(x-data_x[index-1])/2
    return V1+V2+V3+V4



# 绘制三次样条插值后的图像
x_plot = np.linspace(0,52,5000)
y_plot = np.array([MSpline3(x,y,M,h,t) for t in x_plot])
LineLens = 0

# Part2 求曲线长度，使用第一型线积分的策略
# 对曲线长度进行分段求和
for i in range(len(x_plot)-1):
    temp= np.square(x_plot[i+1]-x_plot[i])+np.square(y_plot[i+1]-y_plot[i])
    V = np.sqrt(temp)
    LineLens = LineLens + V

print('***************Lens**********')
print(LineLens)

# 规定横纵轴的单位长度一致，从而使得图像更容易比对和阅读
plt.axis('scaled')
plt.xlim((0,54))
plt.ylim((-15,0))

plt.plot(x_plot,y_plot)
# 描出数据点，设置点打大小和颜色
plt.scatter(x,y,10,'r')
# 中文显示问题解决
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.title('三次样条插值拟合曲线图')

plt.show()

