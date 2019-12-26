import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import struct
# import os

# with open('data20191.dat','rb') as f:
#     data = f.read(4)
#     data_real = struct.unpack_from('f',data)
#     print(data_real)

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
print('--------------header-----------------')
print(Header)

# 构建初始系数矩阵形式
temp = np.array(Header)
temp = temp.reshape((-1))
n = temp[2]
p = temp[3]
q = temp[4]
A = np.zeros((n,n))
b = np.zeros(n)
# 初始化循环参数
RowVa = 0
ColVa = 0
offset = 0
buff = f.read(4)
while buff:
    data = struct.unpack('f',buff)
    buff = f.read(4)
    if RowVa<n:
        A[RowVa,ColVa+offset] = data[0]
    else :
        for i in range(n):
            data = struct.unpack('f',buff)
            b[i] = data[0]
            buff = f.read(4)
            pass
    ColVa += 1 
    if ColVa == p+q+1:
        ColVa = 0
        RowVa += 1 
        if RowVa > q and (RowVa+q)<n : 
            offset += 1

f.close()