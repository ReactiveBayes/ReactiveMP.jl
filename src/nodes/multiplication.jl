
@node typeof(*) Deterministic [out, A, in]

# By default multiplication node uses TinyCorrection() strategy for precision matrix on `in` edge to ensure precision is always invertible
default_meta(::typeof(*)) = TinyCorrection()
