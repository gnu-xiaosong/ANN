# 多分类实例
import pandas as pd
from prettytable import PrettyTable
from prettytable.colortable import ColorTable, Themes
from package_xskj_NetworkXsimple import netGraph
from model import AnnModel
from layers import Dense
import numpy as np


data_set = pd.read_csv('./data/iris_training.csv')

# 数据预处理:注意这里数据处理只能在上面一步把数据类型都转化为数字型的才能进行缺失值判断，因为判断函数仅认数字型!!!!!!!!!!!容易出错
# 1.缺失值处理：替换 or 删除
# data_set.fillna(0, inplace=True)
# 丢弃带有缺失值的数据（只要有一个维度有缺失）
data_set.dropna(how='any', inplace=True)

X = np.array(data_set.iloc[:, :-1])
Y = np.array(data_set.iloc[:, -1]).reshape(-1, 1)
X_labels = data_set.columns.tolist()[:-1]


# 实例化一个神经网络模型
ann = AnnModel()

# 构建一个神经网络
ann.Sequential([
    # 隐藏层1
    Dense(neuron_num=5, active_fun="relu"),
    Dense(neuron_num=10, active_fun="relu"),
    # 输出层
    Dense(neuron_num=3, active_fun="softmax")

])
# 编译模型
## 参数定义
lr=0.01
_lambda=0.2
epochs = 100000
validation_split=0.2
Y_classify=['品种一', '品种二', '品种三']  # 对应Y类别，与索引一一对应

ann.compiler(
    lossType="sparse_categorical_crossentropy",
    gradientAlgorithm="mini_batch_gradient_descent",  # batch_gradient_descent、mini_batch_gradient_descent、stochastic_gradient_descent
    lr=lr,
    validation_splite=validation_split,
    _lambda=_lambda,
    miniBatchSize=32,
    patience=1000,
    Y_classify=Y_classify)


# 数据集分割
train_set, test_set = ann.data_set_splite((X, Y), validation_split=validation_split)

# 模型训练
W_opt, B_opt, min_cost = ann.fit(train_set, epochs=epochs)

print("traing result:")
table_fit = ColorTable(theme=Themes.OCEAN, field_names=['second_by_epoch', 'lr','iteration','regularized paremeter','feature count', 'train_data percent','min cost', 'opt_W', 'opt_b'])
table_fit.add_row([ann.second_enmu_iteration, lr,   epochs,  _lambda,   train_set['X'].shape[1],     str((1-validation_split)*100) + "%" , f'cost={round(min_cost, 5)}'  ,  '---',   '---'])
print(table_fit)



# 模型评价
correct_rate, predict_class, pre_arr, Y_, correct_count, err_count,predict_label= ann.evaluate(test_data=test_set)
print("model evaluate:")
table_evaluate = PrettyTable(['accuracy','correct count','error count'])
table_evaluate.add_row([ round(correct_rate, 4),correct_count,err_count])
print(table_evaluate)

# print(f"真实值: {Y_} 预测值: {pre_arr}")

# 模型预测
predict_set={
    "X": np.array(train_set['X'][:10, :]),
    "Y": train_set['Y'][:10]
}
# 模型预测

predict_y,predict_label = ann.predict(predict_set)

print(f"model predict result: {predict_label}")

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
