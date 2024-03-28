import numpy as np

def dot_bracket_to_matrix(dot_bracket):
    """
    Computes an adjacency matrix from the dot-bracket notation.

    Args:
        dot_bracket: String representing the dot-bracket notation.

    Returns:
        The adjacency matrix representing the design.
    """
    matrix = np.zeros((len(dot_bracket), len(dot_bracket)))
    i = 0
    while i < len(dot_bracket):
        if dot_bracket[i] == ')':
            j = i
            while dot_bracket[j] != '(':
                j -= 1
            matrix[i][j] = 1
            matrix[j][i] = 1
            dot_bracket = dot_bracket[:i] + '.' + dot_bracket[i+1:]
            dot_bracket = dot_bracket[:j] + '.' + dot_bracket[j+1:]
        i += 1
    return matrix

def sparse_to_matrix(sparse_matrix):
    matrix = np.zeros((100, 100))
    for site1, site2 in sparse_matrix:
        matrix[site1, site2] += 1
        matrix[site2, site1] += 1
    return matrix

def mask(input_seq):
    """
    Masks the given sequence randomly with Ns.

    Args:
        input_seq: The sequence to mask

    Returns:
        The masked sequence.
    """
    output = ""
    masking_prob = 0.2
    rand = np.random.binomial(n=1, p=masking_prob, size=len(input_seq))
    for i in range(len(input_seq)):
        if rand[i]:
            output += "N"
        else:
            output += input_seq[i]
    return output

def replace_x(sequence, min_length, max_length):
    num_xs = sequence.count("X")
    mean = (max_length + min_length) / (2 * num_xs)
    # chosen such that in 99.8% of cases, the sum will lie in the interval
    std_dev = 1/3 * (max_length - min_length) / np.sqrt(num_xs)
    new_seq = ""
    
    while True:
        # generate random numbers
        # first mean is the middle of the interval
        # the numbers should add up to, divided by the amount of random variables
        random_N_lens = [mean] * num_xs
        random_N_lens += np.random.randn(num_xs) * std_dev
        random_N_lens = [round(length) for length in random_N_lens]
        if min_length <= sum(random_N_lens) <= max_length:
            break
    for char in sequence:
        if char == "X":
            num_ns = random_N_lens.pop()
            new_seq += "N" * num_ns
            continue
        new_seq += char
    return new_seq

def custom_hamming(input_seq, target_design, folded_mutation):
    return sum(
        [
            input_seq == "N" and
            not (folded == target or
            folded in ['(', ')'] and target == '|')
            for i, (folded, target) in
            enumerate(zip(folded_mutation, target_design))
        ]
    )