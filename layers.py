"""
desc: layers层模块, 操作细度为neuron神经元实体
核心函数Dense()
A_in  ==========> A_out
      compute()
"""
from neuron import  Neuron

def Dense(neuron_num=1, active_fun="linear"):
    """
    desc: 设置layer神经网络层
    paremeters:
        A_in     np.dnumpy   输入列向量
        neuron_num   int     神经元数量
        active_fun   str     激活函数类型 默认linear 可选 sigmod、softmax、rule
    return:
        neuron_arr   list    输出该层神经元实体列表
    """
    # 存储该layer层的神经元
    neuron_arr = []

    for i in range(neuron_num):
        # 实例化一个神经元
        neuron  = Neuron(active_fun=active_fun)
        # 添加进该数组中
        neuron_arr.append(neuron)

    return  neuron_arr




