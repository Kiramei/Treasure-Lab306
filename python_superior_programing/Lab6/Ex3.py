# 3. 类的继承设计
# 每一个二维的形状Shape 都有周长和面积两个属性。 矩形(Square) 和圆形(Circle) 都是Shape的一种。
# 	定义这三个类和他们的继承关系，以便计算不同形状的周长和面积。
# 	设计函数compute(obj),能计算不同形状的周长和面积。

import math


class Shape(object):
    def __init__(self):
        self.perimeter = 0
        self.area = 0

    def compute(self):
        # 接口一：计算周长和面积
        pass

    def __str__(self):
        return 'Perimeter: %d, Area: %d' % (self.perimeter, self.area)

    def __repr__(self):
        return str(self)


class Square(Shape):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def compute(self):
        # 继承接口一：计算正方形的周长和面积
        self.perimeter = self.length * 4
        self.area = self.length ** 2


class Circle(Shape):
    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def compute(self):
        # 继承接口一：计算圆的周长和面积
        self.perimeter = 2 * self.radius * math.pi
        self.area = self.radius ** 2 * math.pi


def compute(obj):
    # 通用普适办法，即使新增形状也不需要修改代码
    obj.compute()
    return obj


if __name__ == '__main__':
    s = Square(5)
    c = Circle(3)
    print(compute(s))
    print(compute(c))
