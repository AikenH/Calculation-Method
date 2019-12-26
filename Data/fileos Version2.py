# import os

# with open('data20191.dat','rb') as f:
#     data = f.read(4)
#     data_real = struct.unpack_from('f',data)
#     print(data_real)
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import struct

f = open('data20192.dat','rb')
# 读取文件的标识头
Header = []
for i in range(5):
    buff = f.read(4)
    data = struct.unpack('l',buff)
#     print(data[0])
    Header.append(data)
    pass
# 输出文件的标识头
print('--------------header for array-----------------')
print(Header)

# 构建初始系数矩阵形式
# 收集输出系数矩阵的必要参数
temp = np.array(Header)
temp = temp.reshape((-1))
Version = temp[1]
n = temp[2]
p = temp[3]
q = temp[4]
A = np.zeros((n,n))
b = np.zeros(n)

# 初始化循环参数：对应行列的index 以及offset
# 
RowVa = 0
ColVa = 0
offset = 0

# 版本控制主体循环框架：258非压缩格式、514压缩格式
if Version == 258:
    # 通过buff控制读取文件的光标位置，根据给出的格式，每四个位是一个数字
    # while控制将文件读取完毕
    buff = f.read(4)
    while buff:
        # 用unpack函数解析float存储的2进制结构
        data = struct.unpack('f',buff)
        # 分开稀疏矩阵A和右端向量b的读取
        if RowVa<n : A[RowVa,ColVa] = data[0]
        else : b[ColVa] = data[0]
        # 以下控制光标和矩阵位置的移动
        ColVa += 1
        buff = f.read(4)
        if ColVa == 10:
            ColVa = 0
            RowVa +=1
            pass
        pass
    # 输出结果
    print('--------------A for the 258----------------')
    print(A)
    print('--------------B for the 258-----------------')
    print(b)
elif Version == 514:
    buff = f.read(4)
    while buff:
        data = struct.unpack('f',buff)
        if RowVa<n:
            A[RowVa,ColVa+offset] = data[0]
        else :
            # 直接一次将b全部读完
            for i in range(n):
                data = struct.unpack('f',buff)
                b[i] = data[0]
                buff = f.read(4)
                pass
            
        buff = f.read(4)
        # 控制光标和矩阵位置的移动
        # 在压缩格式之下，着重对矩阵所属位置的偏移进行控制
        ColVa += 1 
        if ColVa == p+q+1:
            ColVa = 0
            RowVa += 1 
            # 主要基于存储格式的话，考虑下带宽对矩阵数据位置的影像
            # 在左端不在移动的顶点和右侧不在一栋的顶点加以控制。
            if (RowVa > q and (RowVa+q)<n):
                offset += 1
                
    print('--------------A for the 514-----------------')
    print(A)
    print('--------------B for the 514-----------------')
    print(b)

def GaussEli(A,b,n):
f.close()
