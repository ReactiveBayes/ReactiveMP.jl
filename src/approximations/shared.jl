
as_vec(d::Float64) = [d] # Extend vectorization to Float
as_vec(something)  = vec(something) # Avoid type-piracy, but better to refactor this

as_mat(d::Float64) = [d;;]
as_mat(mat::AbstractMatrix) = mat

# This function computes cumulative start indices in the tuple `sizes`, note that `()` acts as size 1`
function __starts_at(sizes::Tuple)
    return last(
        reduce(sizes; init = (0, ())) do result, size
            current, accumulated = result
            next = current + prod(size)
            return (next, (accumulated..., current + 1))
        end
    )
end

Base.@propagate_inbounds __as_vec_copyto!(container, start, input::Real)             = container[start] = input
Base.@propagate_inbounds __as_vec_copyto!(container, start, input::AbstractVecOrMat) = copyto!(container, start, input, 1, length(input))

# This function linearizes the `inputs` argument into one (potentially big) vector
# For example (1, 2) becomes `[ 1, 2 ]`
#             ([2, 3], 2) becomes `[ 2, 3, 2 ]`
#             ([2, 3], [ 1.0 0.0; 0.0 1.0 ]) becomes `[ 2.0, 3.0, 1.0, 0.0, 0.0, 1.0 ]`
#             and so on
function __as_vec(inputs::Tuple)
    sizes     = prod.(size.(inputs))
    starts_at = __starts_at(sizes)
    total     = last(starts_at) + prod(last(sizes)) - 1

    T = promote_type(eltype.(inputs)...)
    x = Vector{T}(undef, total)

    foreach(inputs, starts_at) do input, start
        @inbounds __as_vec_copyto!(x, start, input)
    end

    return x
end

# This function extracts elements from the linearized version of inputs from the `__as_vec` function
# In case of `size = ()` we extract a single element a treat it as it would be a Univariate variable
function __splitjoinelement(x::AbstractVector, start, size::Tuple{})
    return x[start]
end

# This function extracts elements from the linearized version of inputs from the `__as_vec` function
# In case of `size = (Int, )` we extract the specified amount of elements as a view and treat it as a Multivariate variable
function __splitjoinelement(x::AbstractVector, start, size::Tuple{Int})
    return view(x, start:(start + prod(size) - 1))
end

# This function extracts elements from the linearized version of inputs from the `__as_vec` function
# In case of `size = (Int, Int)` we extract the specified amount of elements, reshape it into a matrix and treat it as a Matrixvariate variable
function __splitjoinelement(x::AbstractVector, start, size::Tuple{Int, Int})
    return reshape(view(x, start:(start + prod(size) - 1)), first(size), last(size))
end

# This function extracts elements from the linearized version of inputs from the `__as_vec` function all at once
# Essentially this is the `undo` operation for the `__as_vec`
# See also `__splitjoinelement`
function __splitjoin(x::AbstractVector, sizes::Tuple)
    return map(sizes, __starts_at(sizes)) do size, start
        return __splitjoinelement(x, start, size)
    end
end
