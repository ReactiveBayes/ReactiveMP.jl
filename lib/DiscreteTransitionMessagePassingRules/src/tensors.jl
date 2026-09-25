# The tensor algebra of the node's rules: multiplying a tensor by a vector or a tensor
# along some of its axes, and summing those axes out.

corresponding_size(dim, dims, values) = (occurrence = findfirst(==(dim), dims); occurrence === nothing ? 1 : size(values, occurrence))

"""
    multiply_dimensions!(tensor, dims, values)

Multiply `tensor`, elementwise and in place, by `values` along the axes `dims`: `values`' `k`-th
axis runs along `tensor`'s `dims[k]`, and `dims` are increasing, since `values` is reshaped, not
permuted. Promotes both to a common element type first, returning a
new tensor if it had to.
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
    sum_out_dimensions(tensor, dims, values)

Multiply `tensor` by `values` along the axes `dims` and sum those axes out, keeping them as
singletons: the inner product of `tensor` and `values` over `dims`.
"""
sum_out_dimensions(tensor::AbstractArray, dims::NTuple{N, Int}, values::AbstractArray) where {N} =
    sum(multiply_dimensions!(tensor, dims, values); dims = dims)
