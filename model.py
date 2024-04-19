"""
desc: ANN模型

难点： 1.W，B的矩阵数据定义形式  ******* 以及W和B更新形式(W可采用list存储，每一维度为一个layer层， B采用二维矩阵形式）ok
      2.数据集的载入方式  ok
      3.反向传播求解梯度   no
      4.编码性能优化：主要由于采用了粒度化的神经元实体对象的单一循环计算，这里导致了不能使用矩阵形式GPU加速计算，因素计算性能比较低  no
      5.正则优化解决过度拟合问题 ok
      6.多分类softmax激活函数的计算，和rule激活函数函数 no
"""
import numpy as np

import layers
import plot_model


class AnnModel:
    """
    desc: 神经网络模型
    """
    def __init__(self):
        """初始化操作"""
        pass

    def Sequential(self, layer_list=[
        layers.Dense(),
        layers.Dense(),
    ]):
        """
        desc: 定义模型处理流程顺序 前向传播流程  Hidden layer 配置
        parmeters:
            layer_list =    [
                  Dense(),
                  Dense()
                  .....
                ]
        """
        self.layers = layer_list


    def compiler(self, lossType='mean_squared_error',  lr = 0.01, _lambda=0):
        """
        desc: 针对定义好的模型在模拟和前的处理操作：
              比如向量化W，b，X，等前置操作，便于模型训练拟合
              adem优化等操作都在这里
              损失函数类型定义等
              loss     str     采用的损失函数类型： 均方差mean_squared_error、交叉熵   sparse_categorical_crossentropy等
              _lambda  float   正则化系数 默认0，只惩罚W模型参数
              lr      float    学习率
        """

        # 根据Sequential处理流程获取网络模型结构以初始化W，b 调用init_wb方法


        # 设置损失函数类型
        self.lossType = lossType
        # 设置正则系数
        self._lambda = _lambda
        # 设置学习率
        self.lr = lr
        # 其他一些优化操作：如Adam自学习率优化


    def  data_dispose(self, data_set):
        """
        desc: 数据集预处理
        """

        data_set_dispose=data_set

        return data_set_dispose


    def init_wb(self, layers, data_set):
        """
        desc: 初始化模型参数W、b
        提示：模型参数不能全部初始化为0，逻辑回归和线性回归可以，因为根据递推公式知偏导数除了每层的偏置项外其余节点都为同一个，相当于在重复计算同一个结果，一堆特征变成了一个特征（虽然有很多单元，但值都相等）。
        故相当于仅有一个特征，这样神经网络的性能和效果就会大大折扣，因此需要进行随机初始化。
        随机初始化作用：有效防范梯度爆炸或消失，不同的初始化方式根据激活函数而定，以下为常见的激活函数Relu的初始化
        paremeters:
                layers  list
                 >>[
                  Dense(),
                  Dense()
                  .....
                ]
        return:
            W list  初始化参数
            B list  初始化参数
        """
        # 获取数据集的特征个数
        a_in_num = data_set["X"].shape[1]

        # 定义机构类型
        W = []
        B = []

        for l,layer in enumerate(layers):
            # 获取神经元个数
            neuron_num = len(layer)

            #--------------作为演示用：全部初始化为0时-----------
            # 初始化layer层的偏置b
            # b = np.zeros((neuron_num ,1), dtype=np.float64).reshape(-1, 1)
            # # 初始化: 其列项数应为前一层的a输入个数
            # w = np.zeros((neuron_num, a_in_num), dtype=np.float64)

            #----------------随机初始化:relu----------------------
            b = np.random.rand(neuron_num, 1).reshape(-1, 1)
            # 初始化: 其列项数应为前一层的a输入个数
            w = np.random.rand(neuron_num, a_in_num) * np.sqrt(2 / a_in_num)


            # print(f"b dtype={b.dtype} ")

            # append进入W，B数组中
            W.append(w),B.append(b)

            # 更新# 输入a的个数等于该层的w行数
            a_in_num =  int(w.shape[0])



        return W, B


    def learning_rate(self):
        """
        desc: 学习率
        """


        return  self.lr


    def data_set_splite(self, data_set, validation_split):
        """
        desc: 数据集划分=训练集和测试集

        paremeters:
                data_set=(data_X, data_Y)

        return:
             train_set,test_set =  {
            "X":矩阵  行为样本数  列为特征数
            "Y":  列向量
        }
        """
        validation_splite_m = int(np.ceil(np.array(data_set[0]).shape[0] * validation_split))


        train_set = {
            "X":data_set[0][ validation_splite_m:,:],
            "Y":data_set[1][ validation_splite_m:,:]
        }

        test_set  = {
            "X": data_set[0][ :validation_splite_m,:],
            "Y": data_set[1][ :validation_splite_m,:]
        }


        return  train_set,test_set


    def fit(self,data_X, data_Y,  epochs=5, validation_split=0.1):
        """
        desc: 模型训练拟合
        paremeters:
            data_X             dnumpy       训练数据特征:每行为一个样本，列为特征
            data_Y             dnumpy        训练数据真实值: 列向量对应样本的真实值
            epochs             int      迭代次数
            validation_split   float   数据集划分比例
        """
        # 数据集划分
        train_set, self.test_set = self.data_set_splite((data_X, data_Y), validation_split=validation_split)

        # 初始化模型参数W，b
        W, B = self.init_wb(self.layers, self.test_set)

        # 存储迭代的W、b模型参数
        W_arr = []
        B_arr = []
        Cost = []

        for epoch in range(epochs):

            # 获取学习率
            lr = self.learning_rate()

            # 计算当前模型参数W，B下成本cost值
            cost = self.J_wb(train_set, W, B)
            Cost.append(cost)

            # 一次反向传播：用于求解W和b的偏导值即梯度
            dJ_dW, dJ_dB = self.backward(train_set, W, B)


            # print(f"dJ_dW={dJ_dW} dJ_dB={dJ_dB}")


            # 一次模型参数更新:矢量化更新
            W = [ W[l] - lr * dJ_dW[l]  for l in range(len(self.layers)) ]
            B = [ B[l] - lr * dJ_dB[l]  for l in range(len(self.layers)) ]

            # print(f"W = {W} b={B}")
            # 添加进数组中
            W_arr.append(W)
            B_arr.append(B)

            print(f"epoch={epoch}:   cost={cost} ")



        # 获取最优模型参数索引
        min_cost = min(Cost)  # 求列表最小值
        min_idx = Cost.index(min_cost)  # 求最大值对应索引

        # 绘制损失函数
        plot_model.plot_cost(Cost)

        # 获取最优模型参数
        self.W_opt = W_arr[min_idx]
        self.B_opt = B_arr[min_idx]

        return self.W_opt, self.B_opt, min_idx,min_cost
    def forward(self, data_set, W, B):
        """
        desc: 前向传播算法，用于求解样本一次经过神经网路的输出值
        paremeters:
            data_set={
            "X":,
            "Y":
        }
            W      list  按次序每个item元素为一层layer
            B      list  数组  每item为对应一层layer的偏置
        return:
            a_out  matrix  每列为一个样本的预测值y_hat
            Z      list   每个item元素为对应层layer的Z，列向量对应得到的神经元的Z值 Z = [item_1, item_2...item_i...item_L]    item_l = [z_l1, z_l2...z_li..z_lm]  z_li为列向量对应l层的第i个样本神经元的z
            A      list   每个item元素为对应层layer的A，列向量对应得到的神经元的A值 A = [item_1, item_2...item_i...item_L]    item_l = [a_l1, a_l2...a_li..a_lm]  a_li为列向量 对应l层的第i样本神经元的a
        """
        # 存储计算的Z值和A值： A与Z一一对应的
        Z = []  # 每个item元素为对应层layer的Z，列向量对应得到的神经元的Z值 Z = [item_1, item_2...item_i...item_L]    item_l = [z_l1, z_l2...z_li..z_lm]  z_li为列向量对应l层的第i个样本神经元的z
        A = []  # 每个item元素为对应层layer的A，列向量对应得到的神经元的A值 A = [item_1, item_2...item_i...item_L]    item_l = [a_l1, a_l2...a_li..a_lm]  a_li为列向量 对应l层的第i样本神经元的a

        # 2.前向传播 计算Y_hat  预测值矩阵形式
        ## （1）初始化A为数据集特征输入: a_(0) = x_(i) 每列为样本，对应列值a为l层的a值，默认输出层a_0不append进入A中
        a_in = np.array(data_set["X"]).T  # 这里转置了,输入层 为列向量

        # 遍历更新递归输入
        a = a_in


        ## （2）预测值
        for layer_n in range(len(W)):
            """
            desc:一层layer所做的工作
            具体包括：
                1.计算模型值z 
                2.计算激活值a  遍历作为下一层的输入
            """


            # 线性模型: 所得到的z_l = [z_l1, z_l2...z_li..z_lm]  所有样本的l层的z值
            z = self.layers[layer_n][0].Z(a, W[layer_n], B[layer_n])
            Z.append(z)  # 存储layer_n层的z值

            # 激活:更新A作为输入又作为输出
            a = self.layers[layer_n][0].A_out(z)
            A.append(a)  # 存储layer_n层的a值

        # 输出层的a_out = [a_L1, a_L2...a_Li..a_Lm]  每列为一个样本的输出y_hat=a_L=a_out
        a_out = a

        return a_out, Z, A


    def backward(self, data_set, W, B):
        """
        desc: 反向传播算法，求偏导
        所需数据准备：
            1.数据集：data_set = {
                "X": ,
                "Y":
            }
            单个样本：{x_(i), y_(i)}
            2.计算各层的输出a_(l)
            3.真实值：y_(i)
            4.l层的权重参数W_(l)、l-1层Z_(l-1)
             ----->最终要得到：a_(l) * (l+1层的误差项，即对Z_(l)的偏导)

             以上作为前提步骤：通过前向传播计算获得： W、a、Z
        算法步骤：
            For i to m:
                1.set a_0 = x_(i)
                2.perform forward propagation to compute  a_(l) for l=1、2、3....L
                3.Using y_(i),compute u_(l) = a_(l) -y_(i) 最后一项误差即对最后一层的J对Z的偏导
                4.迭代关系式反向传播计算误差即J对各自层Z的偏导
                5.累计偏导误差得到最后的对参数的偏导

            计算最终的偏导，加上正则项即得到最终的偏导值

            利用梯度关系实现参数的更新


        paremeters:
            data_set={
                "X":,
                "Y":
            }
            W  list    按次序每个item元素为一层layer
            B  list    每项为对应一层layer的偏置
        return:
            dJ_dW_sum  list   与模型参数W一一对应
            dJ_dB_sum  list   与模型参数B一一对应
        """
        # 样本数
        m = len(data_set["Y"])
        #神经网络层数：x输出层不算a_0
        L = len(W)


        # 总体样本的误差(即偏导数)的总和:这里与W和B的形式刚好对应上,形式完全一样，全部初始化为0
        dJ_dW_sum = [ np.array(layer_W ) * 0  for layer_W in W]
        dJ_dB_sum = [ np.array(b) * 0 for b in B]

        # print(f"W= {W} B={B}")

        for i in range(m):
            """
            desc: 循环遍历样本进行单个样本的偏导计算，再累计每个样本的偏导值得出最总的损失函数对W和b的偏导值
            """
            #设置第i个x样本（x_i, y_i)
            x = np.array(data_set["X"])[i,:]
            y = np.array(data_set["Y"])[i,:]

            # 1.set a_(0) = x_(i) 为列向量
            a_0 = np.array(x).reshape(-1, 1)

            # 2.计算前向传播计算各层的a_(l) for  l =1, 2,3....L  调用前向传播算法:self.forward()
            ## 封装data_set
            data_ = {
                "X": np.array([x]) ,
                "Y": np.array([y])
            }

            a_out, z, a =self.forward(data_, W, B)

            # 3.输出层：计算的误差和对W和b的偏导值：
            dJ_dz = np.array(a_out - y)                                         # J对z_(L)的偏导--"数"   为列向量

            # 单个样本的dJ_dz的数组， 用于存储各层的损失误差dJ_dz，每个item为层的dJ_dz，层数为倒序的dJ_dZ =  [L的dJ_dz, L-1的dJ_dz,...,1的dJ_dz]
            dJ_dZ = []
            dJ_dZ.append(dJ_dz)

            # print(f"|------------样本数={i+1}---------|")
            # print(f" a = {a}")
            # print(f"最后一层J对z偏导: dJ_dz_L={dJ_dz}")

            # 4.隐藏层：利用递归关系式求向后传播的误差(偏导数)   采用循环遍历(0开始索引)：L-2 to 0  只需要遍历到隐藏层的第2层
            for l in range(L-1, -1, -1):
                """ l层
                desc: 神经网络反向递归求解单个样本的损失函数loss对z偏导值，
                      再根据递推关系反向传播求解出神经网络中全部的dJ_dz值
                      同时求得J对W和b的偏导值dJ_dw、dJ_db
                       为从倒数第二层l-1层开始递减: l-1、l-2、....、1
                                    数组索引递减: l-2、l-3、....、0
                                    其中 l为数组中对应层的数组索引
                """
                dJ_dw = dJ_dz @ np.transpose(a_0 if l == 0 else a[l - 1])  # J对w_(L)的偏导--"数" 这里判断是否为第一层，是，则a[l-1]=a_0，否则按照a[l-1]
                dJ_db = np.array(dJ_dz).reshape(-1, 1)  * 1                # J对b_(L)的偏导--"数"

                # 梯度检测
                # print(f"layer ={l+1} 层, gradient real: {dJ_dw[0]}")
                # print(f"--------所在层数={l+1}---------")
                # print(f"dJ_dz={dJ_dz} dJ_dw={dJ_dw} dJ_db={dJ_db}")

                dJ_dW_sum[l] = np.array(dJ_dW_sum[l]) + dJ_dw  # 增加进对应层的总和dJ_dw
                dJ_dB_sum[l] = np.array(dJ_dB_sum[l]).reshape(-1,1) + dJ_db  # 增加进对应层的总和dJ_dw


                # 递归更新l-1层的误差dJ_dz
                if l !=0:
                    g_z_greadient = self.layers[l - 1][0].g_derivative(z[l - 1])  # 递归关系推导g_z_greadient为列向量
                    dJ_dz = np.transpose(W[l]) @  dJ_dz * g_z_greadient
                    dJ_dZ.append(dJ_dz)


        # 计算损失项的偏导数dCost_dW
        dCost_dW       =  [ (1 / m) * layer for  layer in dJ_dW_sum]
        dRegularize_dW =  [ (1 / m) * self._lambda * w   for w in W ]


        #计算J对W，b的偏导
        dJ_dW =  [ dCost_dW[l] + dRegularize_dW[l] for l in range(L)]
        dJ_dB =  [ (1 / m) * layer for  layer in dJ_dB_sum]


        return  dJ_dW, dJ_dB


    def J_wb(self, data_set, W, B):
        """
        计算样本的平均成本
        paremeters:
            data_set={
            "X":,
            "Y":
        }
            W  list  按次序每个item元素为一层layer
            B  list  数组  每元素为对应一层layer的偏置
        """
        # 损失函数：用于计算单个样本到的损失函数
        # global L

        # 样本个数
        m = len(data_set["Y"])


        # print("-------------forward for J_wb(self, data_set, W, B)--------------------------")
        ## （3） 获取最终神经网络输出结果: 每行为一个样本的神经网络的输出
        Y_hat = self.forward(data_set, W, B)[0]          # 一次前向传播求预测值，列向量，各层的z和a值，列为所有样本
        Y     = data_set["Y"]                            # 真实值，列向量


        # 3.计算平均样本函数 J = 损失 + 正则
        ##  （1） 计算cost
        cost       = self.cost(Y_hat, Y, m)
        ##  (2)  计算正则化项（防止过拟合，以惩罚参数W)
        regularize = self.regularize(self._lambda,m,W)

        J_wb = cost + regularize


        # # -----------------梯度检测---------------
        # from mpmath import mp
        #
        # epsilon = 1e-4
        #
        # W_plus_epsilon  = W.copy()
        # W_minus_epsilon = W.copy()
        #
        #
        # W_plus_epsilon[0][0,0]  += mp.mpf(epsilon)
        #
        #
        # print(f"W = {W} mp.mpf(W[0][0,0]) + mp.mpf(epsilon)={W[0][0,0] + epsilon}")
        #
        #
        # W_minus_epsilon[0][0,0]  -= mp.mpf(epsilon)
        #
        # Y_hat_plus_epsilon  =  self.forward(data_set, W_plus_epsilon, B)[0]
        # Y_hat_minus_epsilon =  self.forward(data_set, W_minus_epsilon, B)[0]
        #
        # # 损失成本
        # plus_cost  = self.cost(Y_hat_plus_epsilon, Y, m) + regularize
        # minus_cost = self.cost(Y_hat_minus_epsilon, Y, m) + regularize
        #
        # print(f"W_plus_epsilon={W_plus_epsilon} W_minus_epsilon={W_minus_epsilon} plus_cost= { plus_cost} minus_cost={minus_cost}")
        #
        #
        # J_wb_approximate = (cost - minus_cost) / epsilon
        #
        # print(f"gradient approximate:{J_wb_approximate}")


        return  J_wb



    def regularize(self, _lambda, m, W):
        """
        desc: 计算正则化项
        paremeers:
            _lambda   float  正则参数
            m         int    样本个数
            W         list       神经网络的参数W
        return:
            regularize  float 计算的正则化项
        """

        regularize = 0

        for layer_w in W:
            # 转为dnumpy对象
            regularize += np.sum(np.array(layer_w)**2)

        regularize *= (_lambda) / (2*m)

        return regularize


    def loss(self,y_hat, y):
        """
        desc: 根据采用的损失函数类型求解单个样本的损失值
        """
        def mean_squared_error(y_hat, y):
            """
            desc：损失函数计算算法，均方差损失函数  适用于线性回归
            paremeters:
                y_hat    float   神经网络预测值， 列向量
                y        float   对应标签的真实值 列向量
            return:
                loss     float   产生的损失
            """
            loss = np.array(y_hat - y) ** 2

            return loss


        def sparse_categorical_crossentropy(y_hat, y):
            """
            desc：损失函数计算算法，交叉熵损失函数  适用于二分类
            paremeters:
                y_hat    float   神经网络预测值
                y        float   对应标签的真实值
            return:
                loss     float   产生的损失
            """
            # 防止出现除以零的情况，将预测概率中的非零值替换为一个极小值
            epsilon = 1e-15
            y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

            loss = -y * np.log(y_hat)-(1-y) * np.log(1-y_hat)


            return loss


        # 1.判断使用的损失函数类型
        if self.lossType == "mean_squared_error":
            loss = mean_squared_error(np.array(y_hat).reshape(-1,1) , np.array(y).reshape(-1, 1))
        elif self.lossType == "sparse_categorical_crossentropy":
            loss = sparse_categorical_crossentropy(np.array(y_hat).reshape(-1,1) , np.array(y).reshape(-1, 1))

        return  loss


    def cost(self, Y_hat, Y, m):
        """
        desc: 用于求解全体样本的平均损失值
        """

        if self.lossType == "mean_squared_error":
            cost = (1 / (2 * m)) * np.sum(self.loss(Y_hat, Y))
        elif self.lossType == "sparse_categorical_crossentropy":
            cost = (1 / m) * np.sum(self.loss(Y_hat, Y))

        return cost


    def predict(self, data_set):
        """
        desc: 用于根据输入数据进行预测  利用已经训练的模型最优参数： W_opt、B_opt
        paremeters:
            data_set={
            "X":,
            "Y":
            }
        """

        predict_y = self.forward(data_set, self.W_opt, self.B_opt)[0]

        return predict_y


    def evaluate(self):
        """
        desc: 用于评估模型: 利用self.test_set数据集来进行测试
        调用前向传播算法: self.forward()求解预测值
        """
        # 真实值
        Y = np.array(self.test_set['Y']).reshape(-1, 1)
        # 预测
        predict = np.array(self.forward(self.test_set, self.W_opt, self.B_opt)[0][0]).reshape(-1,1)

        # 判断评价类型
        if self.lossType=="mean_squared_error":
            """
            desc: 线性回归模型，采用均方根误差RMSE、平均绝对误差MAE
            """
            # 均方根误差RMSE
            RMSE = np.sqrt(np.mean((Y - predict)**2))
            # 平均绝对误差MAE
            MAE =  np.mean(np.abs(Y - predict))

            return RMSE,MAE,predict,Y
        else:
            """
            desc: 分类问题，测试集模型的准确率
            """
            # 决策阈值
            threshold = 0.5
            predict_result = []
            for i in range(len(predict)):
                if predict[i] > threshold:
                    predict_result.append(1)
                else:
                    predict_result.append(0)

            # 比较
            result = np.abs(np.array(predict_result) - Y.T)

            # 统计正确个数
            err_count = np.sum(result)
            correct_count = len(predict) - err_count

            # 正确率
            correct_rate = correct_count / len(predict)

            return correct_rate, predict_result, predict, Y.T.tolist(), correct_count, err_count


    def customize(self,fun):
        """
        desc:用户自定义函数
        paremeters: self 为该类的对象
        """
        fun(self)

