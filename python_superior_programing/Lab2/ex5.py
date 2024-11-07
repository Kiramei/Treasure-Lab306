import os


def read_names():
    original = os.listdir('./attackresult/target')
    return [x[:-7] for x in original]


def create_dict(names: list) -> dict:
    """
    From names, create a dict with names as keys and a dict as values.
    For each name, fetch the data of target, pred, 0.0300, 0.0700, 0.0900.
    The information are stored in the corresponding files.
    :param names: The list of names from `read_names()`
    :return: The dict with the above structure
    """
    result_dict = {}
    for name in names:
        result_unit = {}
        with open('./attackresult/target/' + name + '.target', 'r') as f:
            result_unit['target'] = int(f.readline().strip('target: '))
        with open('./attackresult/no_attack/' + name + '.pred', 'r') as f:
            result_unit['pred'] = int(f.readline())
        for menu in ['0.0300', '0.0700', '0.0900']:
            with open('./attackresult/pgd_normalize/eps' + menu + '/' + name + '.score', 'r') as f:
                result_unit[menu] = int(f.readline())
        # Filter out the data where the model's prediction is wrong
        if result_unit['target'] == result_unit['pred']:
            result_dict[name] = result_unit
    return result_dict


def calculate_accuracy(filtered_dict: dict, raw_len: int) -> list:
    """
    Calculate the accuracy of the model.
    :param filtered_dict: Dict from `create_dict()`
    :param raw_len: Len of the original dict
    :return: The result list of the accuracy
    """
    accuracy = [0, 0, 0, 0]
    dict_for_accuracy = filtered_dict.items()
    filtered_len = len(dict_for_accuracy)
    # Collect the right predictions and the failed attacks
    for name, result_unit in dict_for_accuracy:
        if result_unit['target'] == result_unit['pred']:
            accuracy[0] += 1
        if result_unit['target'] == result_unit['0.0300']:
            accuracy[1] += 1
        if result_unit['target'] == result_unit['0.0700']:
            accuracy[2] += 1
        if result_unit['target'] == result_unit['0.0900']:
            accuracy[3] += 1
    # Calculate the accuracy
    for i in range(1, len(accuracy)):
        accuracy[i] = 1 - accuracy[i] / filtered_len
    accuracy[0] = accuracy[0] / raw_len
    # Format the accuracy
    for i in range(0, len(accuracy)):
        accuracy[i] = '%.3f' % accuracy[i]
    return accuracy


def P2_5() -> list:
    name_list = read_names()
    extracted = create_dict(name_list)
    return calculate_accuracy(extracted, len(name_list))

if __name__ == '__main__':
    print(P2_5())