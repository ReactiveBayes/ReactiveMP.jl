export make_node

@node typeof(dot) Deterministic [ out, in1, in2 ]

# By default dot-product node uses TinyCorrection() strategy for precision matrix on `in1` and `in2` edges to ensure precision is always invertible
default_meta(::typeof(dot)) = TinyCorrection()
