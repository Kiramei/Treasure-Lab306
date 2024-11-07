def P2_1(score: float) -> str:
    """
    A function that judges a student's grade based on the score.
    :param score: The score of the student
    :return: The grade of the student
    """
    # Filter out invalid scores
    if score < 0 or score > 100:
        return 'X'
    # Use a dictionary to store the score range and the corresponding grade
    grade_dict = {
        (90, float('inf')): 'A',
        (80, 90): 'B',
        (70, 80): 'C',
        (60, 70): 'D',
        (0, 60): 'F'
    }
    # Use a for loop to iterate through the dictionary to find the corresponding grade
    for score_range, grade in grade_dict.items():
        if score_range[0] <= score < score_range[1]:
            return grade


if __name__ == '__main__':
    print(P2_1(100))
    print(P2_1(65))
    print(P2_1(77))
    print(P2_1(40))
    print(P2_1(203))
