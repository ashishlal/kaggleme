def cyjaccard2(seq1, seq2):
    cdef set set1 = set(seq1)
    cdef set set2 = set()

    cdef Py_ssize_t length_intersect = 0

    for char in seq2:
        if char not in set2:
            if char in set1:
                length_intersect += 1
            set2.add(char)

    return 1 - (length_intersect / float(len(set1) + len(set2) - length_intersect))
