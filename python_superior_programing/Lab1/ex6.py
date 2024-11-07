numbers = [i for i in range(100, 1000)]
combinations = [i for i in numbers if len(set(str(i))) == 3]
print(combinations)
