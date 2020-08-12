def _correlate_sparse_offsets(input, indices, offsets, values, output):
    for off, val in zip(offsets, values):
        # this loop order optimises cache access, gives 10x speedup
        for i, j in enumerate(indices):
            output[i] += input[j + off] * val
