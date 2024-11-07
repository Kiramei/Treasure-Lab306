# 2. 设计一个Student类管理学生的成绩，包含如下的属性：
# 	name, id: 姓名和学号
# 	english, math, physics：三科成绩
#
# 假设两个实例A，B， 需要支持如下的操作
# A == B： 如果姓名和学号相等，则返回True， 否则返回False
# A > B:   如果A的三科成绩全部大于B的成绩
# A < B:   如果A的三科成绩全部小于B的成绩
#
# 如果要增加计数器的功能, 统计总共实例化了多少个学生？ 该如何做？

num_of_students = 0


class Student(object):
    def __init__(self, name: str, _id: str, english: int, math: int, physics: int):
        self.name = name
        self.id = _id
        self.english = english
        self.math = math
        self.physics = physics
        self.count = 0

    def __eq__(self, other):
        self.count += 1
        return self.name == other.name and self.id == other.id

    def __gt__(self, other):
        self.count += 1
        return self.english > other.english and self.math > other.math and self.physics > other.physics

    def __lt__(self, other):
        self.count += 1
        return self.english < other.english and self.math < other.math and self.physics < other.physics

    def __str__(self):
        return 'Name: %s, ID: %s, English: %d, Math: %d, Physics: %d' % (
            self.name, self.id, self.english, self.math, self.physics)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return self.count

    def __del__(self):
        global num_of_students
        num_of_students -= 1

    def __new__(cls, *args, **kwargs):
        global num_of_students
        num_of_students += 1
        return super().__new__(cls)


if __name__ == '__main__':
    a = Student('しろは', '001', 80, 90, 96)
    _a = Student('しろは', '001', 77, 52, 66)
    b = Student('こはる', '002', 100, 91, 98)
    c = Student('あやね', '003', 66, 66, 66)
    print('The number of students is', num_of_students)
    print(f'a == _a is {a == _a}')
    print(f'a == b is {a == b}')
    if a > b:
        print('しろは is better than こはる')
    else:
        print('こはる is better than しろは')
    if a < c:
        print('しろは is worse than あやね')
    else:
        print('あやね is worse than しろは')
