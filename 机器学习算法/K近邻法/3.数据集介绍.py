# coding:utf-8

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split

# 1.数据集获取
# 1.1 小数据集获取
iris = load_iris()
# print(iris)

# 1.2 大数据集获取
# news = fetch_20newsgroups()
# print(news)

# 2.数据集属性描述
# print("数据集中特征值是:\n", iris.data)
# print("数据集中目标值是:\n", iris["target"])
# print("数据集中特征值名字是:\n", iris.feature_names)
# print("数据集中目标值名字是:\n", iris.target_names)
# print("数据集的描述:\n", iris.DESCR)

# 3.数据可视化
# 3.1 数据类型转换,把数据用DataFrame存储
iris_data = pd.DataFrame(data=iris.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
iris_data["target"] = iris.target


# print(iris_data)


def iris_plot(data, col1, col2):
    sns.lmplot(x=col1, y=col2, data=data, hue="target", fit_reg=False)
    plt.title("鸢尾花数据展示")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()


# iris_plot(iris_data, "Sepal_Length", 'Petal_Width')
iris_plot(iris_data, 'Sepal_Width', 'Petal_Length')


# 4.数据集的划分
# x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=2)
# print("训练集的特征值是:\n", x_train)
# print("测试集的特征值是:\n", x_test)
# print("训练集的目标值是:\n", y_train)
# print("测试集的目标值是:\n", y_test)


# print("训练集的目标值形状:\n", y_train.shape)
# print("测试集的目标值形状:\n", y_test.shape)


# x_train1, x_test1, y_train1, y_test1 = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
# x_train2, x_test2, y_train2, y_test2 = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
# print("测试集的目标值是:\n", y_test1)
# print("测试集的目标值是:\n", y_test2)


















