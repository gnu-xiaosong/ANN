"""
desc: 设置大数据加载时的自定义迭代器
"""
class BatchIterator:
    def  __init__(self,data,iterCall, nextCall):
        """迭代器初始化操作"""
        self.data = data
        self.index = 0
        self.nextCall =nextCall
        self.iterCall = iterCall

    def __iter__(self):
        """返回迭代器对象"""
        self.iterCall(self)
        return self
    def __next__(self):
        """每次调用用执行的操作"""
        if len(self.data) > self.index:
            # 回调函数用户自己操作,参数为当前迭代器对象
            return  self.nextCall(self)
        else:
            # 迭代结束时抛出 StopIteration 异常
            raise StopIteration








# def call(iterator):
#     """回调函数：参数为迭代器对象"""
#     item = iterator.data[iterator.index]
#     iterator.index+=1
#     return item +20
#
#
# if __name__ == '__main__':
#
#     data = [1, 2, 3, 4]
#     for item in BatchIterator(data, call=call):
#         print(item)


