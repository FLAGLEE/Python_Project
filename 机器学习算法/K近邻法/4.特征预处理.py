# coding:utf-8

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

data = pd.read_csv("./dating.txt")
print(data)

# # 1.归一化
# # 1.1 实例化一个转换器
transfer = MinMaxScaler(feature_range=(0, 1))
# # 1.2 调用fit_transfrom方法
minmax_data = transfer.fit_transform(data[['milage', 'Liters', 'Consumtime']])
print("经过归一化处理之后的数据为:\n", minmax_data)

# 2.标准化
# 2.1 实例化一个转换器
# transfer = StandardScaler()
# 2.2 调用fit_transfrom方法
# minmax_data = transfer.fit_transform(data[['milage', 'Liters', 'Consumtime']])
# print("经过标准化处理之后的数据为:\n", minmax_data)
