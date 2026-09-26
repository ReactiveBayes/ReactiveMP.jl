# The tensor algebra of the node's rules: multiplying a tensor by a vector or a tensor
# along some of its axes, and summing those axes out.

corresponding_size(dim, dims, values) = (occurrence = findfirst(==(dim), dims); occurrence === nothing ? 1 : size(values, occurrence))

"""
    multiply_dimensions!(tensor, dims::NTuple{N, Int}, values) -> tensor

Multiply `tensor`, elementwise and in place, by `values` along the axes `dims`: the `k`-th axis of
`values` runs along the axis `dims[k]` of `tensor`, and `dims` must be increasing, since `values`
is reshaped, not permuted. When the element types differ, both are promoted first and a new
tensor is returned; otherwise `tensor` itself.

# Examples

```jldoctest; setup = :(using DiscreteTransitionMessagePassingRules)
julia> DiscreteTransitionMessagePassingRules.multiply_dimensions!([1.0 2.0; 3.0 4.0], (2,), [10.0, 100.0])
2×2 Matrix{Float64}:
 10.0  200.0
 30.0  400.0
```

See also [`sum_out_dimensions`](@ref DiscreteTransitionMessagePassingRules.sum_out_dimensions).
"""
function multiply_dimensions!(tensor::AbstractArray{T, M}, dims::NTuple{N, Int}, values::AbstractArray{T, N}) where {T, M, N}
    tensor .*= reshape(values, ntuple(dim -> corresponding_size(dim, dims, values), M))
    return tensor
end

function multiply_dimensions!(tensor::AbstractArray{T, M}, dims::NTuple{N, Int}, values::AbstractArray{P, N}) where {T, M, N, P}
    S = promote_type(T, P)
    return multiply_dimensions!(convert_paramfloattype(S, tensor), dims, convert_paramfloattype(S, values))
end

"""
    sum_out_dimensions(tensor, dims::NTuple{N, Int}, values) -> Array

Multiply `tensor` by `values` along the axes `dims`, as
[`multiply_dimensions!`](@ref DiscreteTransitionMessagePassingRules.multiply_dimensions!) does
(and so in place, when the element types agree), and sum those axes out, keeping them as
singletons: the inner product of `tensor` and `values` over `dims`.

# Examples

```jldoctest; setup = :(using DiscreteTransitionMessagePassingRules)
julia> DiscreteTransitionMessagePassingRules.sum_out_dimensions([1.0 2.0; 3.0 4.0], (2,), [0.5, 0.5])
2×1 Matrix{Float64}:
 1.5
 3.5
```
"""
sum_out_dimensions(tensor::AbstractArray, dims::NTuple{N, Int}, values::AbstractArray) where {N} =
    sum(multiply_dimensions!(tensor, dims, values); dims = dims)
