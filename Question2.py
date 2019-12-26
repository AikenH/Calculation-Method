import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from scipy.linalg import solve

# 读取文件，计算每日的平均气温
io = r'temperature.xlsx' #注意将表格方在同个文件夹中，不然要写绝对地址
d = pd.read_excel(io,sheet_name='Sheet1',usecols=[1,2],converters={'max':float, 'min':float}) #读取文件
# 求出7，8，9月每天的平均温度，用于后续的曲线最小二乘拟合改写
Average = np.zeros([3,31])
for i in range(30):
    Average[0][i] = (d['max'][i] + d['min'][i])/2
    Average[1][i] = (d['max'][i+31] + d['min'][i+31])/2
    Average[2][i] = (d['max'][i+61] + d['min'][i+61])/2
Average[0][30] = (d['max'][30] + d['min'][30])/2
Average[2][30] = (d['max'][91] + d['min'][91])/2

print(Average)

# 最小2×拟合函数编写。1次，2次，3次
# 这个函数主要是求解出各项拟合函数的系数
def myl(x,y,k=1):
# k=1,y = a + bx,g=[1,x] 
    if k == 1: 
        g = np.ones([x.size,2])
        for i in range(x.size):
            g[i,1] = x[i]
# k=2 y=a+bx+cx*x
    elif k == 2:
        g = np.ones([x.size,3])
        for i in range(x.size):
            g[i,1]=x[i]
            g[i,2]=np.square(x[i])
# k=3 y=a+bx+cx*x+dx*xxx
    elif k == 3:
        g = np.ones([x.size,4])
        for i in range(x.size):
            g[i,1]=x[i]
            g[i,2]=np.square(x[i])
            g[i,3]=pow(x[i],3)
# g*g'a=g'y
    G = np.transpose(g)
    # 调用函数求解简单的线性方程
    y_solve = np.dot(G,y)
    y_solve = np.transpose(y_solve)
    A = np.dot(G,g)
    para = solve(A,y_solve)
#     print(para)
    
    return para 

# 最小二乘法的数值积分，直接拿式子手撕就行了
# 手写出积分后的表达式
def interfunc(x,para,k):
    # 多项式的积分表达式，好些的禁
    if k == 1 :
        ans = para[0]*x + para[1]*x*x*0.5
    elif k == 2:
        ans = para[0]*x + para[1]*x*x*0.5 + para[2]*x*x*x/3
    elif k == 3:
        ans = para[0]*x + para[1]*x*x*0.5 + para[2]*x*x*x/3 + para[3]*x*x*x*x/4
    return ans

# 将各个月的数据输入进行拟合计算
# 通过k选择最小二乘法的次数（阶数）
k = 2
input_x = np.arange(1,32)
para = myl(input_x,Average[0][:],k)

def fucx(x,para,k):
    if k == 1 :
        ans = para[0]+para[1]*x
    elif k == 2:
        ans = para[0]+para[1]*x+para[2]*x*x
    elif k == 3:
        ans = para[0]+para[1]*x+para[2]*x*x+para[3]*x*x*x
    return ans

x_plot = np.linspace(0,31,3000)
y_plot = np.array([fucx(t,para,k) for t in x_plot])

plt.axis('scaled')
plt.xlim((0,31))
plt.ylim((15,40))
plt.scatter(input_x,Average[0][:])
plt.plot(x_plot,y_plot)

Ans = interfunc(31,para,k)-interfunc(1,para,k)
Ans = Ans/31
print(Ans)

