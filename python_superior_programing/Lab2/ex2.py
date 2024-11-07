# 学生成绩表入data1.txt中所示。对学生成绩按学分加权求出综合测评分，Gym(体育)、Eng(英语)、Math(数学)、物理(Phy)的学分分别为1， 2， 3， 2.5，综合测评分的计算公式如下： ， 为课程成绩， 为相应的学分， ，n为成绩个数. 写出函数names，scores = P2_2() 按照综合测评分从高到低的顺序返回学生的姓名及其分数，其中names为名字序列，scores 为分数的序列。分数保留一位小数位。 提示：排序的时候，可以考虑使用分数作为键值，建立字典。
# # name      Gym   Eng   Math  Phy
# xiaoming    85    90    93    88
# xiaohong    75    95    80    90
# xiaojun     95    80    85    70

# Path: Lab2/ex2.py
def P2_2() -> tuple:
    """
    A function that returns the names and scores of the students in descending order of their comprehensive evaluation scores.
    :return: A tuple containing the names and scores of the students in descending order of their comprehensive evaluation scores
    """
    # Open the file and read the data
    with open('data1.txt', 'r') as f:
        data = f.readlines()
    # Use a dictionary to store the name and score of each student
    student_dict = {}
    # Use a for loop to iterate through the data
    for index, line in enumerate(data):
        if index == 0 or line == '\n':
            continue
        # Split the line
        line = line.split()
        # Get the name of the student
        name = line[0]
        # Get the score of the student
        score = 0
        for i in range(1, len(line)):
            score += float(line[i]) * [1, 2, 3, 2.5][i - 1]
        # Add the name and score to the dictionary
        student_dict[name] = score / (1 + 2 + 3 + 2.5)
    # Sort the dictionary by score in descending order
    # lambda x: x[1] means that the key of the dictionary is the score
    student_dict = sorted(student_dict.items(), key=lambda x: x[1], reverse=True)
    # Get the names and scores of the students
    names = [student[0] for student in student_dict]
    scores = ['%.1f' % student[1] for student in student_dict]
    # Return the names and scores of the students
    return names, scores


if __name__ == '__main__':
    names, scores = P2_2()
    print(names)
    print(scores)
