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
    masking_prob = 0.1
    rand = np.random.binomial(n=1, p=masking_prob, size=len(input_seq))
    for i in range(len(input_seq)):
        if rand[i]:
            output += "N"
        else:
            output += input_seq[i]
    return output

def replace_x(sequence, min_length, max_length):
    """
    Replaces all X in the input string by N such that
    the amount of Ns in the final string is
    between min_length and max_length.

    The goal is to sample random numbers that will be the number of Ns.
    I will approximate the uniformly distributed random variables
    by normally distributed random variables with
    mean (max_length + min_length) / (2 * num_xs), which means that the
    expectancy of the sum is exactly in the middle of the interval and
    standard deviation 1/6 * (max_length - min_length) / sqrt(num_xs),
    the goal of which is to have 99.8% of the random variables
    in the interval to not have to resample a lot.
    This is achieved by setting 3*stddev = (max_length - min_length) / 2
    <=> stddev = (max_length - min_length) / 6
    and as the variance of num_xs random variables is sqrt(num_xs) times
    the variance of one random variable, we have to divide by sqrt(num_xs).

    Args:
        sequence (string): The input sequence with Xs
        min_length (int): The minimum amount of Ns to replace the Xs by
        max_length (int): The maximum amount of Ns to replace the Xs by

    Returns:
        string: The modified sequence without X
    """
    num_xs = sequence.count("X")
    mean = (max_length + min_length) / (2 * num_xs)
    # chosen such that in 99.8% of cases, the sum will lie in the interval
    std_dev = 1/6 * (max_length - min_length) / np.sqrt(num_xs)
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

def custom_hamming(target_design, folded):
    """Computes the hamming distance and allows for | (closing or opening brackets).

    Args:
        target_design (string): The target
        folded (string): The predicted design

    Returns:
        int: The hamming distance
    """
    return sum(
        [
            not (folded_site == target_site or
            folded_site in ['(', ')'] and target_site == '|' or
            target_site == "N")
            for i, (folded_site, target_site) in
            enumerate(zip(folded, target_design))
        ]
    )