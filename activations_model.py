"""
desc: 这是激活函数类型
"""
import numpy as np


def sigmod(x):
    """
    desc: sigmod激活函数   一般作为二分类输出层的激活函数
    paremeters:
        Z_wb   float
    """

    g = 1 / (1 + np.exp(-x))

    return g

def rule_function(x):
    """
    Rule function to apply different rules based on input value.
    Args:
        x (float): Input value to apply rules on.
    Returns:
        float: Result based on input value rules.
    """

    return np.maximum(0, x)


def softmax(z):
    """
       计算Softmax函数

       参数：
       z : numpy.ndarray
           输入的实数向量，形状为 (K,)，其中 K 是类别的数量 列向量

       返回值：
       numpy.ndarray
           Softmax函数的输出，形状与输入相同，为每个类别的概率分布 列向量
       """
    # 对输入向量 z 进行数值稳定处理，减去最大值
    z_stable = z - np.max(z, axis=0)

    # 计算指数部分
    exp_z = np.exp(z_stable)

    # 计算分母，即所有指数的和
    sum_exp_z = np.sum(exp_z, axis=0)

    # 计算Softmax函数的输出，即每个类别的概率
    softmax_output = exp_z / sum_exp_z

    return softmax_output



def sigmod_derivative(x):
    """
    desc: sigmod激活函数的导数
    paremeters:
        x   float/dnumpy
    """

    gradient  = sigmod(x) * (1 - sigmod(x))

    return gradient

def rule_derivative(x):
    """
    Rule function to apply different rules based on input value.
    Args:
        x (float): Input value to apply rules on.
    Returns:
        float: Result based on input value rules.
    """
    gradient = np.where(x>0, 1, 0)

    return gradient


def softmax_derivative(z):
    """
    计算Softmax函数的导数

    参数：
    z : numpy.ndarray
        输入的实数向量，形状为 (K,)，其中 K 是类别的数量

    返回值：
    numpy.ndarray
        Softmax函数的导数，形状为 (K, K)，表示每个类别对每个输入的导数
    """
    # 计算Softmax函数的输出
    softmax_output = softmax(z)

    # 计算导数矩阵
    K = len(z)
    softmax_derivative_matrix = np.diag(softmax_output) - np.outer(softmax_output, softmax_output)

    return softmax_derivative_matrix

