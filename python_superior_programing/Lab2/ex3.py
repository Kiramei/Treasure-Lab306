# 3. data2.npy 中有一个int型的list，里面有个别数字是重复的， 请写出函数P2_3()返回序列，包含所有重复的数字。例如，假设只有3和4有重复， 那么返回[3,4]。 注意：不要使用unique、集合类操作函数。如果你找到了某个高级命令，能够一两行就解决问题，请不要使用。写出你的算法！
# 请使用下面的代码读取数据：
# 	import numpy as np
# 	data = list(np.load(‘data2.npy’))
import numpy as np


# Path: Lab2/ex3.py
def P2_3() -> list:
    """
    A function that returns a list containing all the duplicate numbers.
    :return: A list containing all the duplicate numbers
    """
    # Open the file and read the data
    with open('data2.npy', 'rb') as f:
        data = np.load(f)
    # Use a dictionary to store the number and the number of occurrences
    num_dict = {}
    # Use a for loop to iterate through the data
    for num in data:
        # If the number is not in the dictionary, add it to the dictionary
        if num not in num_dict:
            num_dict[num] = 1
        # If the number is in the dictionary, increase the number of occurrences by 1
        else:
            num_dict[num] += 1
    # Use a list to store the duplicate numbers
    duplicate_nums = []
    # Use a for loop to iterate through the dictionary
    for num, count in num_dict.items():
        # If the number of occurrences is greater than 1, add the number to the list
        if count > 1:
            duplicate_nums.append(num)
    # Return the list
    return duplicate_nums

if __name__ == '__main__':
    print(P2_3())