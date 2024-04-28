"""
desc: 图像种类识别案例
"""
import numpy as np
import pandas as pd
from package_xskj_NetworkXsimple import netGraph
from prettytable import PrettyTable
from prettytable.colortable import ColorTable, Themes
from layers import  Dense
from model import AnnModel
from tools import traverse_folder_for_files, unpickle, Z_score_normalization
# 获取描述信息
mate_path = "./data/cifar-10-batches-py/batches.meta"

mate_data = unpickle(mate_path)
print(mate_data)

# 实例化一个神经网络模型
ann = AnnModel()

# 构建一个神经网络
ann.Sequential([
    # 隐藏层1
    Dense(neuron_num=50, active_fun="relu"),
    Dense(neuron_num=10, active_fun="relu"),
    # 输出层
    Dense(neuron_num=10, active_fun="softmax")

])
# 编译模型
## 参数定义
lr=0.01
_lambda=0.2
epochs = 200
validation_split=0.2
Y_classify=list(mate_data[b'label_names'])  # 对应Y类别，与索引一一对应

ann.compiler(
    lossType="sparse_categorical_crossentropy",
    gradientAlgorithm="mini_batch_gradient_descent", # batch_gradient_descent、stochastic_gradient_descent、mini_batch_gradient_descent
    miniBatchSize=80,
    lr=lr,
    _lambda=_lambda,
    patience=10,
    validation_splite=validation_split,
    Y_classify=Y_classify)


# 使用迭代器进行数据集的加载
def nextCall(batchIterator):
    """
    迭代器item处理
    """
    # 读取batch的数据
    file_name = batchIterator.data[batchIterator.index]
    batchIterator.index +=1
    # 读取内容
    data_set, batch_name= unpick_data_set(file_name)
    print(f"batch data_set: {batch_name}")

    return data_set


def iterCall(batchIterator):
    """
    迭代器item处理,枚举之前调用一次
    """
    data = batchIterator.data
    print(f"iterator arr: {data}")



def unpick_data_set(file):
    """
    解刨数据集
    """
    data_item = unpickle(file)
    batch_name = data_item[b"batch_label"]
    X_data = np.array(data_item[b'data']) / 255
    Y_labels = np.array([data_item[b'labels']]).reshape(-1, 1)

    # 数据正则化
    # X_data = Z_score_normalization(X_data)

    # 封装数据集
    data_set = {
        "X": X_data,
        "Y": Y_labels
    }

    return data_set, batch_name



# 定义迭代器初始化传入数据
train_path = "./data/cifar-10-batches-py/train"
test_path  = "./data/cifar-10-batches-py/test"

data_train_path = traverse_folder_for_files(train_path)  # 训练数据
data_test_path  = traverse_folder_for_files(test_path)   # 测试数据


# 测试集
test_set = unpick_data_set(data_test_path[0])[0]

ann.data_iterator(nextCall=nextCall, iterCall=iterCall)

# 特征量数
feature_count = test_set["X"].shape[1]


# 模型训练
W_opt, B_opt, min_cost = ann.fit(
          data_train_path,
          feature_count=feature_count,
          epochs=epochs)

print("traing result:")
table_fit = ColorTable(theme=Themes.OCEAN, field_names=['second_by_epoch', 'lr','iteration','regularized paremeter','feature count', 'train_data percent','min cost', 'opt_W', 'opt_b'])
table_fit.add_row([ann.second_enmu_iteration, lr,   epochs,  _lambda,   feature_count,     str((1-validation_split)*100) + "%" , f'cost={round(min_cost, 5)}'  ,  '---',   '---'])
print(table_fit)

# 模型评价
correct_rate, predict_class,pre_arr, Y, correct_count, err_count,predict_label= ann.evaluate(test_data=test_set)
print("model evaluate:")
table_evaluate = PrettyTable(['accuracy','correct count','error count'])
table_evaluate.add_row([ round(correct_rate, 4),correct_count,err_count])
print(table_evaluate)


# 保存模型参数
ann.save_model_paremeters(test_sample_count=len(test_set["X"]),  test_correct_rate=correct_rate, name="cifar-10-batches")