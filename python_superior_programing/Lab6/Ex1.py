# 1.
# 设计类
# Deque实现双向队列。
#
# 假设类实例为obj，需要实现如下的成员函数：
#     x = obj.pop_back()
# 删除尾部的元素
#     x = obj.pop_front()
# 删除头部的元素
#     obj.push_back(x)
# 在尾部加入一个元素
#     obj.push_front(x)
# 在头部加入一个元素
#     obj.size()
# 返回双向队列中元素的个数
#     obj.extend(B)
# 合并两个队列，B的尾部和obj的头部相连
# 并支持如下的函数：
# 	len(obj): 				返回队列的长度
# 	obj[k]: 				支持下标操作，返回第k个元素

class Deque(object):
    def __init__(self):
        self.items = []
    def isEmpty(self):
        return self.items == []
    def pop_back(self):
        return self.items.pop()
    def pop_front(self):
        return self.items.pop(0)
    def push_back(self, item):
        self.items.append(item)
    def push_front(self, item):
        self.items.insert(0, item)
    def size(self):
        return len(self.items)
    def extend(self, B):
        self.items.extend(B.items)
    def __str__(self):
        return str(self.items)
    def __len__(self):
        return len(self.items)
    def __getitem__(self, key):
        return self.items[key]


if __name__ == '__main__':
    # Demo for usage of Deque
    d = Deque()
    print(d.isEmpty())
    d.push_back(4)
    d.push_front('dog')
    print('Length is: ', len(d))
    d.push_front('cat')
    print('d[2] is: ', d[2])
    d.push_back(True)
    print(d.size())
    print(d.isEmpty())
    d.push_back(8.4)
    print(d.pop_front())
    print(d.pop_back())
