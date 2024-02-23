def read_eterna():
    dot_brackets = []
    with open("eterna100_vienna2.txt", 'r') as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            dot_bracket = line.split('\t')[4]
            dot_brackets.append(dot_bracket)
    return dot_brackets


def read_train_data(data_dir=None):
    file_path = "data/train.fasta"
    if data_dir is not None:
        file_path = data_dir + file_path
    dot_brackets = []
    with open(file_path, "r") as file:
        for line in file:
            if not line.startswith(">"):
                dot_brackets.append(line)
    return dot_brackets


def read_test_data(data_dir=None):
    file_path = "data/test.fasta"
    if data_dir is not None:
        file_path = data_dir + file_path
    dot_brackets = []
    with open(file_path, "r") as file:
        for line in file:
            if not line.startswith(">"):
                dot_brackets.append(line)
    return dot_brackets


def read_validation_data(data_dir=None):
    file_path = "data/valid.fasta"
    if data_dir is not None:
        file_path = data_dir + file_path
    dot_brackets = []
    with open(file_path, "r") as file:
        for line in file:
            if not line.startswith(">"):
                dot_brackets.append(line)
    return dot_brackets


def filter_data(dot_brackets, max_length):
    return [dot_bracket for dot_bracket in dot_brackets if len(dot_bracket) <= max_length]
