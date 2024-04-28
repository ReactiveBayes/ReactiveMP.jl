export dot

import LinearAlgebra: dot

@node typeof(dot) Deterministic [out, in1, in2]

# By default dot-product node uses `MatrixCorrectionTools.ReplaceZeroDiagonalEntries(tiny)` strategy for precision matrix on `in1` and `in2` edges to ensure precision is always invertible
default_meta(::typeof(dot)) = MatrixCorrectionTools.ReplaceZeroDiagonalEntries(tiny)
