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
    ("C", "="): 3,
    ("C", "N"): 4,
    ("A", "."): 5,
    ("A", "("): 6,
    ("A", ")"): 7,
    ("A", "="): 8,
    ("A", "N"): 9,
    ("U", "."): 10,
    ("U", "("): 11,
    ("U", ")"): 12,
    ("U", "="): 13,
    ("U", "N"): 14,
    ("G", "."): 15,
    ("G", "("): 16,
    ("G", ")"): 17,
    ("G", "="): 18,
    ("G", "N"): 19,
    ("=", "."): 20,
    ("=", "("): 21,
    ("=", ")"): 22,
    ("=", "="): 23,
    ("=", "N"): 24,
    ("N", "."): 25,
    ("N", "("): 26,
    ("N", ")"): 27,
    ("N", "="): 28,
    ("N", "N"): 29
}

single_bracket_to_int = {
    ("C", "."): 0,
    ("C", "("): 1,
    ("C", ")"): 2,
    ("C", "|"): 3,
    ("C", "="): 4,
    ("C", "N"): 5,
    ("A", "."): 6,
    ("A", "("): 7,
    ("A", ")"): 8,
    ("A", "|"): 9,
    ("A", "="): 10,
    ("A", "N"): 11,
    ("U", "."): 12,
    ("U", "("): 13,
    ("U", ")"): 14,
    ("U", "|"): 15,
    ("U", "="): 16,
    ("U", "N"): 17,
    ("G", "."): 18,
    ("G", "("): 19,
    ("G", ")"): 20,
    ("G", "|"): 21,
    ("G", "="): 22,
    ("G", "N"): 23,
    ("=", "."): 24,
    ("=", "("): 25,
    ("=", ")"): 26,
    ("=", "|"): 27,
    ("=", "="): 28,
    ("=", "N"): 29,
    ("N", "."): 30,
    ("N", "("): 31,
    ("N", ")"): 32,
    ("N", "|"): 33,
    ("N", "="): 34,
    ("N", "N"): 35
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
        for index, symbol in enumerate(target, 0):
            if symbol == "(":
                stack.append(index)
            elif symbol == ")":
                paired_site = stack.pop()
                pairing_encoding[paired_site] = index
                pairing_encoding[index] = paired_site
        return pairing_encoding
    
def encode_pairing(target):
    pairing_encoding = [None] * len(target)
    stack = []
    for index, symbol in enumerate(target, 0):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")" and stack:
            paired_site = stack.pop()
            pairing_encoding[paired_site] = index
            pairing_encoding[index] = paired_site
        elif symbol == "N":
            stack = []
    return pairing_encoding

def probabilistic_pairing(target):
    #TODO: Maybe infer from closing brackets
    count = 0
    pairing_encoding = [None] * len(target)
    stack = []
    for index, symbol in enumerate(target, 0):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")" and stack:
            paired_site = stack.pop()
            pairing_encoding[paired_site] = index
            pairing_encoding[index] = paired_site
        elif symbol == "N" and stack:
            stack.pop()
    prob = 0