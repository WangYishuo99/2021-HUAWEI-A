import csv
import numpy as np
from scipy.io import loadmat
# 读取数据
def read_data(filepath):
    matrix = []
    with open(filepath,'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            line_matrix = []
            for line_data in line:
                line_data = line_data.replace('i','j')
                myoutput = complex(line_data)
                line_matrix.append(myoutput)
            matrix.append(line_matrix)
    return matrix

# 读取.mat数据
def read_data_mat():
    data = []
    data1 = loadmat('V1.mat')
    data2 = loadmat('V2.mat')
    data3 = loadmat('V3.mat')
    data4 = loadmat('V4.mat')
    data5 = loadmat('V5.mat')
    data6 = loadmat('V6.mat')
    data.append(data1['data1'])
    data.append(data2['data2'])
    data.append(data3['data3'])
    data.append(data4['data4'])
    data.append(data5['data5'])
    data.append(data6['data6'])
    return data


