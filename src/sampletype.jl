# select_inner_type(::Type{T}) where {T} = eltype(T)
sampletype(::Any) = error("sampletype for arbitrary input is not implemented")
sampletype(dist::Distribution) = eltype(dist)
sampletype(::SampleList{D, S, W, C, M}) where {D, S, W, C, M} = eltype(W)
