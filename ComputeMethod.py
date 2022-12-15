from ReadData import *
import numpy as np
import pandas as pd
import openpyxl as op
import os
import xlwt
import scipy.io

# 读取.mat文件，并将矩阵储存起来
# data_num表示第几个表
# row表示表中矩阵的行数
# col表示表中矩阵的列数
# 返回表中所有矩阵
def read_mat_file(data_num):
    data = np.array(read_data_mat()[data_num])
    # 用于存储整个矩阵
    matrix_V = []
    for i in range(4):
        # 用于存储矩阵的每一行
        matrix_V_line = []
        for j in range(384):
            V_ij = data[i*64:(i+1)*64, j*2:(j+1)*2]
            matrix_V_line.append(V_ij)
        matrix_V.append(matrix_V_line)
    return np.array(matrix_V)

# 获取v_k的方法
# data_num表示第几个表
# K表示第几列的Vk，不大于383，0表示第一列
def get_v_k(data_num, K):
    data = read_mat_file(data_num)
    v_k = data[0][K]
    for i in range(1,4):
        v_k = np.hstack((v_k,data[i][K]))
    return v_k

# 计算矩阵的共轭转置
def conj_t(A):
    return np.conj(np.array(A)).T

# 奇异值分解
def svd(matrix):
    svd_matrix = []
    u,d,v = np.linalg.svd(np.array(matrix))
    svd_matrix.append(u)
    svd_matrix.append(d)
    svd_matrix.append(v)
    return svd_matrix

# 利用svd求解线性方程组Ax=b
def solve_equations_by_svd(A,b):
    b = np.array(b)
    svd_matrix = svd(A)
    u,d,v = svd_matrix[0],svd_matrix[1],svd_matrix[2]
    u = np.array(u)
    d = np.array(d)
    v = np.array(v)
    y = np.ones([np.shape(v)[0],1])
    b = np.matmul(conj_t(u),b)
    for i in range(np.shape(y)[0]):
        y[i] = b[i] / d[i]
    return np.matmul(conj_t(v),y)

# 简单的矩阵乘法实现A*B
def matrix_multiply(A, B):
    res = [[0] * len(B[0]) for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                res[i][j] += A[i][k] * B[k][j]
    return res

# 矩阵乘常数
def matrix_multiply_a(matrix_a,a):
    shape_A = np.shape(matrix_a)
    for i in range(shape_A[0]):
        for j in range(shape_A[1]):
            matrix_a[i][j] *= a
    return matrix_a

# 简单的矩阵相加
def matrix_add(matrix_a, matrix_b):
    rows = len(matrix_a)
    columns = len(matrix_a[0])
    matrix_c = [list() for i in range(rows)]
    for i in range(rows):
        for j in range(columns):
            matrix_c_temp = matrix_a[i][j] + matrix_b[i][j]
            matrix_c[i].append(matrix_c_temp)
    return matrix_c

# 简单的矩阵相减
def matrix_minus(matrix_a, matrix_b):
    rows = len(matrix_a)
    columns = len(matrix_a[0])
    matrix_c = [list() for i in range(rows)]
    for i in range(rows):
        for j in range(columns):
            matrix_c_temp = matrix_a[i][j] - matrix_b[i][j]
            matrix_c[i].append(matrix_c_temp)
    return matrix_c

# 矩阵的分块
def matrix_divide(matrix_a, row, column):
    length = len(matrix_a)
    matrix_b = [list() for i in range(length // 2)]
    k = 0
    for i in range((row - 1) * length // 2, row * length // 2):
        for j in range((column - 1) * length // 2, column * length // 2):
            matrix_c_temp = matrix_a[i][j]
            matrix_b[k].append(matrix_c_temp)
        k += 1
    return matrix_b

# 矩阵的拼接
def matrix_merge(matrix_11, matrix_12, matrix_21, matrix_22):
    length = len(matrix_11)
    matrix_all = [list() for i in range(length * 2)]
    for i in range(length):
        matrix_all[i] = list(matrix_11[i]) + list(matrix_12[i])
    for j in range(length):
        matrix_all[length + j] = list(matrix_21[j]) + list(matrix_22[j])
    return matrix_all

# Strassen算法计算矩阵的乘法
def strassen(matrix_a, matrix_b):
    rows = len(matrix_a)
    if rows == 1:
        matrix_all = [list() for i in range(rows)]
        matrix_all[0].append(matrix_a[0][0] * matrix_b[0][0])
    elif rows <= 16:
        matrix_all = matrix_multiply(matrix_a, matrix_b)
    else:
        s1 = matrix_minus((matrix_divide(matrix_b, 1, 2)), (matrix_divide(matrix_b, 2, 2)))
        s2 = matrix_add((matrix_divide(matrix_a, 1, 1)), (matrix_divide(matrix_a, 1, 2)))
        s3 = matrix_add((matrix_divide(matrix_a, 2, 1)), (matrix_divide(matrix_a, 2, 2)))
        s4 = matrix_minus((matrix_divide(matrix_b, 2, 1)), (matrix_divide(matrix_b, 1, 1)))
        s5 = matrix_add((matrix_divide(matrix_a, 1, 1)), (matrix_divide(matrix_a, 2, 2)))
        s6 = matrix_add((matrix_divide(matrix_b, 1, 1)), (matrix_divide(matrix_b, 2, 2)))
        s7 = matrix_minus((matrix_divide(matrix_a, 1, 2)), (matrix_divide(matrix_a, 2, 2)))
        s8 = matrix_add((matrix_divide(matrix_b, 2, 1)), (matrix_divide(matrix_b, 2, 2)))
        s9 = matrix_minus((matrix_divide(matrix_a, 1, 1)), (matrix_divide(matrix_a, 2, 1)))
        s10 = matrix_add((matrix_divide(matrix_b, 1, 1)), (matrix_divide(matrix_b, 1, 2)))
        p1 = strassen(matrix_divide(matrix_a, 1, 1), s1)
        p2 = strassen(s2, matrix_divide(matrix_b, 2, 2))
        p3 = strassen(s3, matrix_divide(matrix_b, 1, 1))
        p4 = strassen(matrix_divide(matrix_a, 2, 2), s4)
        p5 = strassen(s5, s6)
        p6 = strassen(s7, s8)
        p7 = strassen(s9, s10)
        c11 = matrix_add(matrix_add(p5, p4), matrix_minus(p6, p2))
        c12 = matrix_add(p1, p2)
        c21 = matrix_add(p3, p4)
        c22 = matrix_minus(matrix_add(p5, p1), matrix_add(p3, p7))
        matrix_all = matrix_merge(c11, c12, c21, c22)
    return matrix_all

# 基于Strassen算法的求逆算法
def inv_strassen(matrix_a):
    rows = len(matrix_a)
    if rows >= 2:
        if rows <= 4 or rows % 2 == 1:
            matrix_all = np.linalg.inv(matrix_a)
        else:
            M1 = inv_strassen(matrix_divide(matrix_a, 1, 1))
            M2 = strassen(matrix_divide(matrix_a, 2, 1), M1)
            M3 = strassen(M1, matrix_divide(matrix_a, 1, 2))
            M4 = strassen(matrix_divide(matrix_a, 2, 1), M3)
            M5 = matrix_minus(M4, matrix_divide(matrix_a, 2, 2))
            M6 = inv_strassen(M5)
            C12 = strassen(M3, M6)
            C21 = strassen(M6, M2)
            M7 = strassen(M3, C21)
            C11 = matrix_minus(M1, M7)
            C22 = np.array(M6).dot(-1)
            matrix_all = matrix_merge(C11, C12, C21, C22)
        return matrix_all
    else:
        return np.linalg.inv(matrix_a)


# 基于Strassen算法的求逆优化算法
def inv_strassen_neo(matrix_a):
    matrix_a = np.array(matrix_a)
    rows = len(matrix_a)
    cols = len(matrix_a[0])
    if (rows <= 4) or (rows % 2 == 1 or cols % 2 == 1):
        matrix_a_np = np.array(matrix_a)
        matrix_all = np.linalg.inv(matrix_a_np)
    else:
        M1 = inv_strassen_neo(matrix_divide(matrix_a, 1, 1))
        M2 = strassen(matrix_divide(matrix_a, 2, 1), M1)
        M3 = conj_t(M2)
        M4 = strassen(matrix_divide(matrix_a, 2, 1), M3)
        M5 = matrix_minus(M4, matrix_divide(matrix_a, 2, 2))
        M6 = inv_strassen_neo(M5)
        C12 = strassen(M3, M6)
        C21 = conj_t(C12)
        M7 = strassen(M3, C21)
        C11 = matrix_minus(M1, M7)
        C22 = np.array(M6).dot(-1)
        matrix_all = matrix_merge(C11, C12, C21, C22)
    return matrix_all

# Coppersmith-Winograd算法
def coppersmith_winograd(matrix_a, matrix_b):
    rows = len(matrix_a)
    if rows == 1:
        matrix_all = [list() for i in range(rows)]
        matrix_all[0].append(matrix_a[0][0] * matrix_b[0][0])
    elif rows <= 16:
        matrix_all = np.matmul(matrix_a, matrix_b)
    else:
        S1 = matrix_add(matrix_divide(matrix_a, 2, 1),matrix_divide(matrix_a, 2, 2))
        S2 = matrix_minus(S1,matrix_divide(matrix_a, 1, 1))
        S3 = matrix_minus(matrix_divide(matrix_a, 1, 1),matrix_divide(matrix_a, 2, 1))
        S4 = matrix_minus(matrix_divide(matrix_a, 1, 2),S2)
        T1 = matrix_minus(matrix_divide(matrix_b, 1, 2),matrix_divide(matrix_b, 1, 1))
        T2 = matrix_minus(matrix_divide(matrix_b, 2, 2),T1)
        T3 = matrix_minus(matrix_divide(matrix_b, 2, 2),matrix_divide(matrix_b, 1, 2))
        T4 = matrix_minus(T2,matrix_divide(matrix_b, 2, 1))

        M1 = coppersmith_winograd(matrix_divide(matrix_a, 1, 1),matrix_divide(matrix_b, 1, 1))
        M2 = coppersmith_winograd(matrix_divide(matrix_a, 1, 2),matrix_divide(matrix_b, 2, 1))
        M3 = coppersmith_winograd(S4,matrix_divide(matrix_b, 2, 2))
        M4 = coppersmith_winograd(matrix_divide(matrix_a, 2, 2),T4)
        M5 = coppersmith_winograd(S1,T1)
        M6 = coppersmith_winograd(S2,T2)
        M7 = coppersmith_winograd(S3,T3)

        U1 = matrix_add(M1,M2)
        U2 = matrix_add(M1,M6)
        U3 = matrix_add(U2,M7)
        U4 = matrix_add(U2,M5)
        U5 = matrix_add(U4,M3)
        U6 = matrix_minus(U3,M4)
        U7 = matrix_add(U3,M5)
        matrix_all = matrix_merge(U1, U5, U6, U7)
    return matrix_all

# 基于Coppersmith-Winograd算法的求逆算法
def inv_coppersmith_winograd(matrix_a):
    matrix_a = np.array(matrix_a)
    rows = len(matrix_a)
    cols = len(matrix_a[0])
    if (rows <= 4) or (rows % 2 == 1 or cols % 2 == 1):
        matrix_a_np = np.array(matrix_a)
        matrix_all = np.linalg.inv(matrix_a_np)
    else:
        M1 = inv_coppersmith_winograd(matrix_divide(matrix_a, 1, 1))
        M2 = coppersmith_winograd(matrix_divide(matrix_a, 2, 1),M1)
        M3 = coppersmith_winograd(M1,matrix_divide(matrix_a, 1, 2))
        M4 = coppersmith_winograd(matrix_divide(matrix_a, 2, 1),M3)
        M5 = matrix_minus(M4,matrix_divide(matrix_a, 2, 2))
        M6 = inv_coppersmith_winograd(M5)
        C12 = coppersmith_winograd(M3,M6)
        C21 = coppersmith_winograd(M6,M2)
        M7 = coppersmith_winograd(M3,C21)
        C11 = matrix_minus(M1,M7)
        C22 = np.array(M6).dot(-1)
        matrix_all = matrix_merge(C11, C12, C21, C22)
    return matrix_all

# 计算Wk的方法
# 将Wk拆分后存储到一个列表中
def compute_w_k(matrix_v_k):
    list_w_k = []
    matrix_v_k_mul = np.matmul(conj_t(matrix_v_k), matrix_v_k)
    o2i = matrix_multiply_a(np.eye(len(matrix_v_k_mul)), 0.01)
    add = matrix_add(matrix_v_k_mul, o2i)
    u = inv_coppersmith_winograd(add)
    w_k = np.array(np.matmul(matrix_v_k, u))
    for i in range(4):
        list_w_k.append(w_k[:,i*2:(i+1)*2])
    return list_w_k

# 批量计算Wk的方法
# data_one_base = data_all_six_base[i]
# W_ij = data_one_base[j][i]
def compute_all_w_k():
    # 用来储存所有6个数据库的矩阵
    data_all_six_base = []
    # 读取6个.mat文件
    for data_num in range(6):
        # 用于储存一个数据库中的数据
        data_one_base = []
        # 数据库的第几列(即第几个V_K)
        for col in range(384):
            # 用于储存数据库的一列Wk
            data_col = []
            v_k = get_v_k(data_num,col)
            list_w_k = compute_w_k(v_k)
            for i in range(4):
                data_col.append(list_w_k[i])
            data_one_base.append(data_col)
        data_all_six_base.append(data_one_base)
    return data_all_six_base

# 将data_one_base中的数据组成一个大数组
def combine_all_matrix(data_one_base):
    list_all = []
    for i in range(4):
        list_row = np.array(data_one_base[0][i])
        for j in range(1,384):
            list_row = np.hstack((list_row,np.array(data_one_base[j][i])))
        list_all.append(list_row)
    matrix_1 = np.vstack((np.array(list_all[0]),np.array(list_all[1])))
    matrix_2 = np.vstack((np.array(list_all[2]),np.array(list_all[3])))
    matrix = np.vstack((matrix_1,matrix_2))
    return np.array(matrix)

# 将所有数据输出
def output():
    data_all_six_base = compute_all_w_k()
    for k in range(6):
        data_one_base_matrix = combine_all_matrix(data_all_six_base[k])
        matrix = {'mat': data_one_base_matrix}
        scipy.io.savemat(os.getcwd() + '\\data' + str(k+1) + '.mat', matrix)
