import random

design_to_int = {
    ".": 0,
    "(": 1,
    ")": 2,
    "=": 3
}

seq_design_to_int = {
    ("C", "."): 0,
    ("C", "("): 1,
    ("C", ")"): 2,
    ("C", "["): 3,
    ("C", "]"): 4,
    ("C", "="): 5,
    ("C", "N"): 6,
    ("A", "."): 7,
    ("A", "("): 8,
    ("A", ")"): 9,
    ("A", "["): 10,
    ("A", "]"): 11,
    ("A", "="): 12,
    ("A", "N"): 13,
    ("U", "."): 14,
    ("U", "("): 15,
    ("U", ")"): 16,
    ("U", "["): 17,
    ("U", "]"): 18,
    ("U", "="): 19,
    ("U", "N"): 20,
    ("G", "."): 21,
    ("G", "("): 22,
    ("G", ")"): 23,
    ("G", "["): 24,
    ("G", "]"): 25,
    ("G", "="): 26,
    ("G", "N"): 27,
    ("=", "."): 28,
    ("=", "("): 29,
    ("=", ")"): 30,
    ("=", "["): 31,
    ("=", "]"): 32,
    ("=", "="): 33,
    ("=", "N"): 34,
    ("N", "."): 35,
    ("N", "("): 36,
    ("N", ")"): 37,
    ("N", "["): 38,
    ("N", "]"): 39,
    ("N", "="): 40,
    ("N", "N"): 41
}

single_bracket_to_int = {
    ("C", "."): 0,
    ("C", "("): 1,
    ("C", ")"): 2,
    ("C", "|"): 3,
    ("C", "["): 4,
    ("C", "]"): 5,
    ("C", "="): 6,
    ("C", "N"): 7,
    ("A", "."): 8,
    ("A", "("): 9,
    ("A", ")"): 10,
    ("A", "|"): 11,
    ("A", "["): 12,
    ("A", "]"): 13,
    ("A", "="): 14,
    ("A", "N"): 15,
    ("U", "."): 16,
    ("U", "("): 17,
    ("U", ")"): 18,
    ("U", "|"): 19,
    ("U", "["): 20,
    ("U", "]"): 21,
    ("U", "="): 22,
    ("U", "N"): 23,
    ("G", "."): 24,
    ("G", "("): 25,
    ("G", ")"): 26,
    ("G", "|"): 27,
    ("G", "["): 28,
    ("G", "]"): 29,
    ("G", "="): 30,
    ("G", "N"): 31,
    ("=", "."): 32,
    ("=", "("): 33,
    ("=", ")"): 34,
    ("=", "|"): 35,
    ("=", "["): 36,
    ("=", "]"): 37,
    ("=", "="): 38,
    ("=", "N"): 39,
    ("N", "."): 40,
    ("N", "("): 41,
    ("N", ")"): 42,
    ("N", "|"): 43,
    ("N", "["): 44,
    ("N", "]"): 45,
    ("N", "="): 46,
    ("N", "N"): 47
}

def encode_dot_bracket(dot_bracket, rna_seq, state_radius):
    padding = "=" * state_radius
    dot_bracket = padding + dot_bracket + padding
    if rna_seq is None:
        return [design_to_int[site] for site in dot_bracket]
    rna_seq = padding + rna_seq + padding
    if "|" in rna_seq:
        return [single_bracket_to_int[site] for site in zip(rna_seq, dot_bracket)]
    return [seq_design_to_int[site] for site in zip(rna_seq, dot_bracket)]

def encode_pairing_without_N(target):
        pairing_encoding = [None] * len(target)
        stack = []
        pseudo_knots = []
        for index, symbol in enumerate(target, 0):
            if symbol == "(":
                stack.append(index)
            elif symbol == "[":
                pseudo_knots.append(index)
            elif symbol == ")":
                paired_site = stack.pop()
                pairing_encoding[paired_site] = index
                pairing_encoding[index] = paired_site
            elif symbol == "]" and pseudo_knots:
                paired_site = pseudo_knots.pop()
                pairing_encoding[paired_site] = index
                pairing_encoding[index] = paired_site
        return pairing_encoding
    
def encode_pairing(target):
    pairing_encoding = [None] * len(target)
    stack = []
    pseudo_knots = []
    for index, symbol in enumerate(target, 0):
        if symbol == "(":
            stack.append(index)
        elif symbol == "[":
            pseudo_knots.append(index)
        elif symbol == ")" and stack:
            paired_site = stack.pop()
            pairing_encoding[paired_site] = index
            pairing_encoding[index] = paired_site
        elif symbol == "]" and pseudo_knots:
            paired_site = pseudo_knots.pop()
            pairing_encoding[paired_site] = index
            pairing_encoding[index] = paired_site
        elif symbol == "N":
            stack = []
            pseudo_knots = []
    return pairing_encoding

def probabilistic_pairing(target):
    opening_brackets_until_i = []
    opening_brackets_after_i = []
    closing_brackets_until_i = []
    closing_brackets_after_i = []
    n_until_i = []
    n_after_i = []

    # count how many closing brackets come before/after a certain point and
    # how many opening brackets come before/after a certain point
    count_opening = 0
    count_closing = 0
    n_idx = set()
    for index, symbol in reversed(list(enumerate(target, 0))):
        if symbol == "(":
            count_opening += 1
        if symbol == ")":
            count_closing += 1
        if symbol == "N":
            n_idx.add(index)
        opening_brackets_after_i = [count_opening] + opening_brackets_after_i
        closing_brackets_after_i = [count_closing] + closing_brackets_after_i
        n_after_i = [set(n_idx)] + n_after_i

    # if at some site with an N, there is a discrepancy between the
    # opening brackets and following closing brackets,
    # set a closing bracket with a certain probability
    indices_set = set()
    for index, symbol in reversed(list(enumerate(target, 0))):
        opening_after = opening_brackets_after_i[index]
        closing_after = closing_brackets_after_i[index]
        to_close = opening_after - closing_after - len(indices_set)
        if to_close > 0:
            possible_n = list(n_after_i[index] - indices_set)
            n_chosen = random.sample(possible_n, to_close)
            list_target = list(target)
            for i in n_chosen:
                list_target[i] = ")"
            target = "".join(list_target)
            indices_set.update(n_chosen)

    # vice-versa
    count_opening = 0
    count_closing = 0
    n_idx = set()
    for index, symbol in enumerate(target, 0):
        if symbol == "(":
            count_opening += 1
        if symbol == ")":
            count_closing += 1
        if symbol == "N":
            n_idx.add(index)
        opening_brackets_until_i.append(count_opening)
        closing_brackets_until_i.append(count_closing)
        n_until_i.append(set(n_idx))

    indices_set = set()
    for index, symbol in enumerate(target, 0):
        opening_before = opening_brackets_until_i[index]
        closing_before = closing_brackets_until_i[index]
        to_open = closing_before - opening_before - len(indices_set)
        if to_open > 0:
            possible_n = list(n_until_i[index] - indices_set)
            n_chosen = random.sample(possible_n, to_open)
            list_target = list(target)
            for i in n_chosen:
                list_target[i] = "("
            target = "".join(list_target)
            indices_set.update(n_chosen)
    
    # replace the remaining Ns
    target = target.replace("N", ".")

    # now we have a valid target and call the method without N
    return encode_pairing_without_N(target)
