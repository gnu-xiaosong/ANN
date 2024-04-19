import pandas as pd
from prettytable import PrettyTable
from prettytable.colortable import ColorTable, Themes
from package_xskj_NetworkXsimple import netGraph
from model import AnnModel
from layers import Dense
import numpy as np

# 加载数据集
column_names = [
    'Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
    'Uniformity of Cell Shape', 'Marginal Adhesion',
    'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
    'Normal Nucleoli', 'Mitoses', 'Class'
]
data_set = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
    names=column_names)

# object转float
data_set["Bare Nuclei"] = pd.to_numeric(data_set["Bare Nuclei"],errors='coerce')

# 数据预处理:注意这里数据处理只能在上面一步把数据类型都转化为数字型的才能进行缺失值判断，因为判断函数仅认数字型!!!!!!!!!!!容易出错
# 1.缺失值处理：替换 or 删除
data_set.fillna(0,inplace=True)
# 丢弃带有缺失值的数据（只要有一个维度有缺失）
# data_set.dropna(how='any', inplace=True)


# 将标签替换成0 或 1
min_value = data_set.iloc[:, -1].min()
max_value = data_set.iloc[:, -1].max()
data_set.iloc[:, -1].replace([min_value,max_value], [0,1], inplace=True)

# 删掉编号列
data_set.drop('Sample code number', axis=1, inplace=True)

X = np.array(data_set.iloc[:, :-1])
Y = np.array(data_set.iloc[:, -1]).reshape(-1, 1)
X_labels = data_set.columns





# 实例化一个神经网络模型
ann = AnnModel()

# 构建一个神经网络
ann.Sequential([
    # 隐藏层1
    Dense(neuron_num=15, active_fun="relu"),
    # Dense(neuron_num=10, active_fun="relu"),
    # 输出层
    Dense(neuron_num=1, active_fun="sigmod")

])
# 编译模型
## 参数定义
lr=0.01
_lambda=0.1
epochs = 1000
validation_split=0.2

ann.compiler(lossType="sparse_categorical_crossentropy", lr=lr, _lambda=_lambda)


# 模型训练
W_opt, B_opt, min_idx,min_cost = ann.fit(X, Y, epochs=epochs, validation_split=validation_split)

print("traing result:")
table_fit = ColorTable(theme=Themes.OCEAN, field_names=['lr','iteration','regularized paremeter','feature count', 'train_data percent','min cost', 'opt_W', 'opt_b'])
table_fit.add_row([lr,   epochs,  _lambda,   X.shape[1],     str((1-validation_split)*100) + "%" , f'index={min_idx} cost={round(min_cost, 5)}'  ,  '---',   '---'])
print(table_fit)



# 模型评价
correct_rate, predict_result, predict,y,correct_count,err_count= ann.evaluate()
print("model evaluate:")
table_evaluate = PrettyTable(['accuracy','correct count','error count'])
table_evaluate.add_row([ round(correct_rate, 4),correct_count,err_count])
print(table_evaluate)

# 模型预测
data_set_pre={
    "X":X[0, :],
    "Y":Y[0, :]
    }
# 模型预测
predict = ann.predict(data_set_pre)
print(f"model predict result: {predict}")

# 用户自定义函数
def customize(net):
    """
    desc: 用户自定义函数
    paremeter： ann为该实例对象
    """
    # 绘制神经网络图
    # 实例化netGraph对象
    networkGraph = netGraph(type=1)
    input_layer = []

    # 增加输入层
    for n in  range(len(X_labels)):
        # 设置节点在网络中的位置
        pos = (1, n+1)
        # name名
        name = "$x_{%d}$" % (n+1)
        input_layer.append({
            "name": name,
            "pos":pos
        })
        # label设置
        label = X_labels[n]

        # 设置节点
        networkGraph.addNode(
            name=name,
            pos=pos,
            label=label
        )


    # 增加隐藏层和输出层
    for layer_n in range(len(net.layers)):
        """
        desc:遍历神经网络的结构层
        """
        # 遍历该层中的节点
        for n in range(len(net.layers[layer_n])):
            # 设置节点在网络中的位置
            pos = (layer_n+2, n + 1)
            # name名
            name = "$a^{(%d)}_{%d}$" % (layer_n + 2, n+1)
            # label设置
            # label = X_labels[n]
            # 增加网络节点
            networkGraph.addNode(
                name=name,
                pos=pos,
                # label="该节点node的描述label",
                # label_color="label的颜色，默认black",
                # 出度edge边信息
                # nexts=[
                #     {
                #         "node": "连接node的name",
                #         "label": "edge边标签",
                #         "color": "edge边标签颜色",
                #         "weight": edge边的权重
                #     },
                # ],
                previous=
                [
                    {
                        "node": node['name'],
                        "label": "",
                        "color": "blue",
                        "weight": 1
                    }
                    for node in input_layer
                ]
                if layer_n == 0
                else
                [
                    # 其它层
                    {
                        "node":"$a^{(%d)}_{%d}$" % (layer_n + 1, n+1),
                        # "label": "",
                        "color": "red",
                        "weight": 1
                    }
                    for n in range(len(net.layers[layer_n-1]))
                ]
            )
    # 绘制网络图
    networkGraph.draw()


ann.customize(customize)






