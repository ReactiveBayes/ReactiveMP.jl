# The additive coupling layer.

@doc raw"""
    AdditiveCouplingLayer(flow; partition_dim::Int = 1, permute::Bool = true)

A layer of a [`FlowModel`](@ref) that adds to each coordinate a coupling flow of the one before
it. For an input ``x \in \mathbb{R}^d``,

```math
y_1 = x_1, \qquad y_k = x_k + f_{k-1}(x_{k-1}), \quad k = 2, \dots, d,
```

with a flow ``f_{k-1}`` of its own for each ``k``, all of the kind of `flow`. It is invertible
whatever the flows are, ``x_k = y_k - f_{k-1}(x_{k-1})`` in order, and its Jacobian is
lower-triangular with a unit diagonal, so its determinant is one. For ``d = 2`` it is the layer
of Dinh, Krueger and Bengio, "NICE: Non-linear independent components estimation",
arXiv:1410.8516 (2014).

!!! warning "Scalar partitions only"
    Only `partition_dim = 1` works: every coordinate is its own partition. A larger
    `partition_dim` builds a model whose [`FlowMessagePassingRules.forward`](@ref) throws a
    `MethodError`, since a coupling flow is not applied to a block of coordinates.

# Arguments

- `flow`: the coupling flow placeholder, [`PlanarFlow`](@ref)`()` or [`RadialFlow`](@ref)`()`.
  The layer's dimension is set when it is wrapped in a [`FlowModel`](@ref), and the flows'
  parameters by [`compile`](@ref).

# Keywords

- `partition_dim`: the size of a partition, which must divide the model's dimension. Default
  `1`, the only size that works.
- `permute`: whether the model places a random [`PermutationLayer`](@ref) after the layer, so
  that the next coupling layer mixes the coordinates in another order. Default `true`.

# Examples

```jldoctest; setup = :(using FlowMessagePassingRules)
julia> model = compile(FlowModel(3, (AdditiveCouplingLayer(PlanarFlow(); permute = false),)), zeros(6));

julia> FlowMessagePassingRules.forward(model, [1.0, 2.0, 3.0]) ≈ [1.0, 3.0, 5.0]   # u = 0: each fₖ(x) = x
true
```
"""
struct AdditiveCouplingLayer{T <: NTuple{N, AbstractCouplingFlow} where {N}} <:
    AbstractCouplingLayer
    dim::Int
    f::T
    partition_dim::Int
end
struct AdditiveCouplingLayerEmpty{
        T <: NTuple{N, AbstractCouplingFlowEmpty} where {N},
    } <: AbstractCouplingLayer
    dim::Int
    f::T
    partition_dim::Int
end
struct AdditiveCouplingLayerPlaceholder{T <: AbstractCouplingFlowEmpty, B} <:
    AbstractLayerPlaceholder
    f::T
    partition_dim::Int
    permute::Val{B}
end
function AdditiveCouplingLayer(
        flow::T; partition_dim::Int = 1, permute::Bool = true
    ) where {T <: AbstractCouplingFlowPlaceholder}
    return AdditiveCouplingLayerPlaceholder(
        prepare(partition_dim, flow), partition_dim, Val(permute)
    )
end

# include permute as value type for type stability; the permutation is drawn from `rng`
_prepare(dim::Int, layer::AdditiveCouplingLayerPlaceholder) = _prepare(default_rng(), dim, layer)
function _prepare(
        rng::AbstractRNG, dim::Int, layer::AdditiveCouplingLayerPlaceholder{T, true}
    ) where {T}
    ## TODO: generalize for non-1 partition dim and overlap
    @assert dim % getpartitiondim(layer) == 0 "The input dimensionality is not exactly divisible by the partition dimension."
    nr_maps = (dim ÷ getpartitiondim(layer)) - 1
    maps = ntuple((x) -> getf(layer), Val(nr_maps))
    return AdditiveCouplingLayerEmpty(dim, maps, getpartitiondim(layer)),
        PermutationLayer(rng, dim)
end
function _prepare(
        rng::AbstractRNG, dim::Int, layer::AdditiveCouplingLayerPlaceholder{T, false}
    ) where {T}
    ## TODO: generalize for non-1 partition dim and overlap
    @assert dim % getpartitiondim(layer) == 0 "The input dimensionality is not exactly divisible by the partition dimension."
    nr_maps = (dim ÷ getpartitiondim(layer)) - 1
    maps = ntuple((x) -> getf(layer), Val(nr_maps))
    return AdditiveCouplingLayerEmpty(dim, maps, getpartitiondim(layer))
end

# compile layer (tuples are compiled according to compile in flow_model.jl)
compile(layer::AdditiveCouplingLayerEmpty) = compile(default_rng(), layer)
compile(rng::AbstractRNG, layer::AdditiveCouplingLayerEmpty) = AdditiveCouplingLayer(getdim(layer), compile(rng, getf(layer)), getpartitiondim(layer))
compile(layer::AdditiveCouplingLayerEmpty, params) = AdditiveCouplingLayer(getdim(layer), compile(getf(layer), params), getpartitiondim(layer))

# calculates the number of parameters in the model
nr_params(layer::AdditiveCouplingLayer) = mapreduce(nr_params, +, getf(layer))
nr_params(layer::AdditiveCouplingLayerEmpty) = mapreduce(nr_params, +, getf(layer))

# get-functions for the AdditiveCouplingLayer structure
getf(layer::AdditiveCouplingLayer) = layer.f
getflow(layer::AdditiveCouplingLayer) = layer.f
getdim(layer::AdditiveCouplingLayer) = layer.dim
getpartitiondim(layer::AdditiveCouplingLayer) = layer.partition_dim

# get-functions for the AdditiveCouplingLayerPlaceholder structure
getf(layer::AdditiveCouplingLayerPlaceholder) = layer.f
getflow(layer::AdditiveCouplingLayerPlaceholder) = layer.f
getpartitiondim(layer::AdditiveCouplingLayerPlaceholder) = layer.partition_dim

# get-functions for the AdditiveCouplingLayerEmpty structure
getdim(layer::AdditiveCouplingLayerEmpty) = layer.dim
getf(layer::AdditiveCouplingLayerEmpty) = layer.f
getflow(layer::AdditiveCouplingLayerEmpty) = layer.f
getpartitiondim(layer::AdditiveCouplingLayerEmpty) = layer.partition_dim

# custom Base function for the AdditiveCouplingLayer structure
Base.eltype(layer::AdditiveCouplingLayer{T}) where {T} =
    promote_type(map(eltype, getf(layer))...)

# forward pass through the additive coupling layer
function _forward(layer::AdditiveCouplingLayer, input::AbstractVector{<:Real})

    result = similar(input)

    forward!(result, layer, input)

    return result
end
forward(layer::AdditiveCouplingLayer, input::AbstractVector{<:Real}) =
    _forward(layer, input)
Broadcast.broadcasted(
    ::typeof(forward),
    layer::AdditiveCouplingLayer,
    input::AbstractVector{<:AbstractVector{<:Real}},
) = broadcast(_forward, Ref(layer), input)

# inplace forward pass through the additive coupling layer
function forward!(
        output::AbstractVector{<:Real},
        layer::AdditiveCouplingLayer,
        input::AbstractVector{<:Real},
    )

    f = getf(layer)
    dim = getdim(layer)
    pdim = getpartitiondim(layer)

    @assert length(input) == dim "The dimensionality of the AdditiveCouplingLayer does not correspond to the length of the passed input/output."

    # optimized version for scalar partition dimension
    return if pdim == 1
        output[1] = input[1]
        for k in 2:dim
            output[k] = input[k]
            output[k] += forward(f[k - 1], input[k - 1])
        end
    else
        view(output, 1:pdim) .= view(input, 1:pdim)
        for k in 2:(dim ÷ pdim)
            view(output, (1 + (k - 1) * pdim):(k * pdim)) .= view(
                input, (1 + (k - 1) * pdim):(k * pdim)
            )
            view(output, (1 + (k - 1) * pdim):(k * pdim)) .+= forward(
                f[k - 1], view(input, (1 + (k - 2) * pdim):((k - 1) * pdim))
            )
        end
    end
end

# backward pass through the additive coupling layer
function _backward(layer::AdditiveCouplingLayer, output::AbstractVector{<:Real})

    result = similar(output)

    backward!(result, layer, output)

    return result
end
backward(layer::AdditiveCouplingLayer, output::AbstractVector{<:Real}) =
    _backward(layer, output)
Broadcast.broadcasted(
    ::typeof(backward),
    layer::AdditiveCouplingLayer,
    output::AbstractVector{<:AbstractVector{<:Real}},
) = broadcast(_backward, Ref(layer), output)

# inplace backward pass through the additive coupling layer
function backward!(
        input::AbstractVector{<:Real},
        layer::AdditiveCouplingLayer,
        output::AbstractVector{<:Real},
    )

    f = getf(layer)
    dim = getdim(layer)
    pdim = getpartitiondim(layer)

    @assert length(input) == dim "The dimensionality of the AdditiveCouplingLayer does not correspond to the length of the passed input/output."

    return if pdim == 1
        input[1] = output[1]
        for k in 2:dim
            input[k] = output[k]
            input[k] -= forward(f[k - 1], input[k - 1])
        end
    else
        view(input, 1:pdim) .= view(output, 1:pdim)
        for k in 2:(dim ÷ pdim)
            view(input, (1 + (k - 1) * pdim):(k * pdim)) .= view(
                output, (1 + (k - 1) * pdim):(k * pdim)
            )
            view(input, (1 + (k - 1) * pdim):(k * pdim)) .-= forward(
                f[k - 1], view(input, (1 + (k - 2) * pdim):((k - 1) * pdim))
            )
        end
    end
end

# jacobian of the additive coupling layer
function _jacobian(
        layer::AdditiveCouplingLayer, input::AbstractVector{T}
    ) where {T <: Real}

    dim = getdim(layer)

    Ti = promote_type(eltype(layer), T)
    result = zeros(Ti, dim, dim)

    jacobian!(result, layer, input)

    return LowerTriangular(result)
end
jacobian(layer::AdditiveCouplingLayer, input::AbstractVector{<:Real}) =
    _jacobian(layer, input)
Broadcast.broadcasted(
    ::typeof(jacobian),
    layer::AdditiveCouplingLayer,
    input::AbstractVector{<:AbstractVector{<:Real}},
) = broadcast(_jacobian, Ref(layer), input)

# inplace jacobian through the additive coupling layer
function jacobian!(
        result::AbstractMatrix{T},
        layer::AdditiveCouplingLayer,
        input::AbstractVector{<:Real},
    ) where {T <: Real}

    f = getf(layer)
    dim = getdim(layer)
    pdim = getpartitiondim(layer)

    @assert length(input) == dim "The dimensionality of the AdditiveCouplingLayer does not correspond to the length of the passed input/output."

    result .= zero(T)
    for k in 1:(dim ÷ pdim)
        result[k, k] = one(T)
    end
    for k in 1:(dim ÷ pdim - 1)
        result[(1 + k * pdim):((k + 1) * pdim), (1 + (k - 1) * pdim):(k * pdim)] .+= jacobian(
            f[k], input[(1 + (k - 1) * pdim):(k * pdim)]
        )
    end
    return
end

# inverse jacobian of the additive coupling layer
function _inv_jacobian(
        layer::AdditiveCouplingLayer, output::AbstractVector{T}
    ) where {T <: Real}

    dim = getdim(layer)

    Ti = promote_type(eltype(layer), T)
    result = zeros(Ti, dim, dim)

    inv_jacobian!(result, layer, output)

    return LowerTriangular(result)
end
inv_jacobian(layer::AdditiveCouplingLayer, output::AbstractVector{<:Real}) =
    _inv_jacobian(layer, output)
Broadcast.broadcasted(
    ::typeof(inv_jacobian),
    layer::AdditiveCouplingLayer,
    output::AbstractVector{<:AbstractVector{<:Real}},
) = broadcast(_inv_jacobian, Ref(layer), output)

# inplace inv_jacobian through the additive coupling layer
function inv_jacobian!(
        result::AbstractMatrix{T},
        layer::AdditiveCouplingLayer,
        output::AbstractVector{<:Real},
    ) where {T <: Real}

    f = getf(layer)
    dim = getdim(layer)
    pdim = getpartitiondim(layer)

    @assert length(output) == dim "The dimensionality of the AdditiveCouplingLayer does not correspond to the length of the passed input/output."

    # calculate input of layer for simpler jacobian calculation
    input = backward(layer, output)

    result .= zero(T)
    for k in 1:(dim ÷ pdim)
        result[k:end, k] .= one(T)
    end
    for k in 1:(dim ÷ pdim - 1)
        result[(k + 1):end, 1:k] .*=
            -jacobian(f[k], input[(1 + (k - 1) * pdim):(k * pdim)])
    end
    return
end

# extra utility functions
det_jacobian(layer::AdditiveCouplingLayer, input::AbstractVector{<:Real}) = 1.0
absdet_jacobian(layer::AdditiveCouplingLayer, input::AbstractVector{<:Real}) = 1.0
logdet_jacobian(layer::AdditiveCouplingLayer, input::AbstractVector{<:Real}) = 0.0
logabsdet_jacobian(layer::AdditiveCouplingLayer, input::AbstractVector{<:Real}) = 0.0

detinv_jacobian(layer::AdditiveCouplingLayer, output::AbstractVector{<:Real}) = 1.0
absdetinv_jacobian(layer::AdditiveCouplingLayer, output::AbstractVector{<:Real}) = 1.0
logdetinv_jacobian(layer::AdditiveCouplingLayer, output::AbstractVector{<:Real}) = 0.0
logabsdetinv_jacobian(layer::AdditiveCouplingLayer, output::AbstractVector{<:Real}) = 0.0
