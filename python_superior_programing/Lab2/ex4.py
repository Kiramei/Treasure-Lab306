# 4. 文本文件text.txt的内容是一篇短文。写出函数P2_4(filename)以元组的形式返回内容中频率最高的5个单词(频率由高到低的顺序)。该函数能测试不同的文件，因此其输入参数是文件名。忽略所有的标点符号，区分大小写。请按照如下的架构实现该函数：
# def P2_4(filename)
# lines = readAFile(filename)
# dict = lines2WordDict(lines)
# result = processDict(dict)
# return result
#
# 提示：对字典的内容进行排序
# 按key进行排序；
# d2 = {x : d1[x] for x in sorted(d1.keys())}
# 按value 进行排序：
# d2 = dict(sorted(d1.items(), key=lambda x : x[1], reverse=False))


# Path: Lab2/ex4.py

def readAFile(filename: str) -> list:
    with open(filename, 'r') as f:
        return f.readlines()

def lines2WordDict(lines: list) -> dict:
    word_dict = {}
    for line in lines:
        line = line.split()
        for word in line:
            # Ignore all punctuation marks
            word = word.strip(',.?!;:')
            # Ignore all empty strings
            if word == '':
                continue
            # Add unknown words to the dictionary
            if word not in word_dict:
                word_dict[word] = 1
            # Increase the number of occurrences of known words by 1
            else:
                word_dict[word] += 1
    return word_dict

def processDict(word_dict: dict) -> list:
    # Sort the dictionary by the number of occurrences in descending order
    word_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
    # Return the 5 most frequent words
    return [word[0] for word in word_dict[:5]]
def P2_4(filename: str) -> list:
    """
    A function that returns the 5 most frequent words in the content in the form of a tuple.
    :param filename: The name of the file
    :return: The 5 most frequent words in the content in the form of a tuple
    """
    # Read the file
    lines = readAFile(filename)
    # Convert the lines to a dictionary
    _dict = lines2WordDict(lines)
    # Process the dictionary
    result = processDict(_dict)
    # Return the 5 most frequent words
    return result

if __name__ == '__main__':
    print(P2_4('text.txt'))