'''
通过对矩阵的最终结果的分析，我认为最后的结果应该是一个稳定值，但是由于高斯消去法的舍去误差

'''
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import struct
# import scipy.linalg

# 文件读取，可以通过这里选择文件读取
# 对所有的文件都可以进行处理，改变一个数字即可
# 解析文件头，并输出
filename = 'data20194.dat'
Headinfo = np.fromfile(filename,dtype=np.int32)
head = Headinfo[0:5].astype(np.int32)
print(head)
print('``````````````````````head↑```````````````````````````')
# 获取文件头以及一些基本的数据参量
Version = head[1]
n = head[2]
p = head[3]
q = head[4]
offset =[]
# 针对带状矩阵的压缩存储格式，给出了（i，j）中不同i
# 所会带来的矩阵横向坐标上的偏差
for i in range(n):
    if i>q:offset.append(i-q)
    else: offset.append(0)
offset = np.array(offset)

# 读取矩阵数据
data = np.fromfile(filename,dtype=np.float32)
# 针对不同格式的矩阵进行不同类型的读取
# 压缩格式的带状矩阵的0值在后续消去的时候进行判断后填充
if Version == 258:
    start = 5
    end = n*n+5
    A = data[start:end].astype(np.float32)
    b = data[end:end+n].astype(np.float32)
    A = A.reshape((n,n))
    pass
elif Version == 514:
    start = 5
    end = n*(p+q+1)+5
    A = data[start:end].astype(np.float32)
    b = data[end:end+n].astype(np.float32)
    num = p+q+1
    A = A.reshape((n,num))

    # 对压缩型的带状矩阵进行重构，重组其压缩形式，将所有需要填充的0都放在右侧
    # 以此用来匹配offset的坐标偏差
    for i in range(p):
        index = -1
        # 找到第一个非0 元所在的index
        for j in range(num):
            if A[i][j] !=0 and index == -1: 
                index = j
        # 重构矩阵进行替换
        if index != -1:
            for k in range(num-index):
                A[i][k] = A[i][k+index]
            A[i][num-index:num] = 0

print(A)
print('````````````````````reshpae A↑`````````````````````````````')
print(b)
print('````````````````````reshape b↑`````````````````````````````')

# 接下来编写高斯消去法（带状矩阵专用）
# 消去的过程中只需要有限的取值就行

def GaussEli(A,b,Version,n,q,p,offset):
    if Version == 258: offset[:]=0
    '''高斯消去法，编写上没有什么需要注意的地方，就是普通的逐行进行消去叭了
    通过我们对压缩格式的重组，我们可以将非压缩格式的坐标也变换成普通格式，所以不需要重复编写
    但是要注意到的是，对于带状矩阵，需要计算的只有两个方向上的p和q个单位长度就行了，
    由于带外元素都是0，多余的计算是没必要进行的'''
    for index in range(n):
        for i in range(1,q+1):
            if index+i<n:
                # 先求出l
                l = A[index+i,index-offset[index+i]]/A[index,index-offset[index]]
                for j in range(p+1):
                    if index+j<n: 
                        # 逐行进行消去化简
                        A[index+i,index+j-offset[index+i]] -= A[index,index+j-offset[index]]*l
                        # 对右侧参数也进行变换
                b[index+i] -= b[index]*l
    # 上面得到了A，b消去以后的矩阵，后面通过回带求解x[]
    x = np.zeros(n)
    # 回代过程的编写
    # 统一的形式 b-（a？*x？）/aii，然后通过反向计算即可
    # 由于初始化的x value是0，所以用同一的形式反向回代就可以了
    # 需要注意的是在编程的过程中不要让矩阵的index越界就可以了
    for i in range(n-1,-1,-1):
        sumA = 0
        for j in range(p):
            index = i+j
            if index<n: sumA += A[i,index-offset[i]]*x[index]
            
        for j in range(q):
            index = i-j
            if index>0: sumA += A[i,index-offset[i]]*x[index]

        x[i] = round((b[i]-sumA)/A[i,i-offset[i]],5)
    # 输出结果
    print(x)
    print('`````````````````````result↑````````````````````````````')
    print(A)
    print('`````````````````````gauss a↑````````````````````````````')
    print(b)
    print('`````````````````````gauss b↑````````````````````````````')
    print(offset)
    print('``````````````````````offset↑```````````````````````````')

    pass

GaussEli(A,b,Version,n,q,p,offset)