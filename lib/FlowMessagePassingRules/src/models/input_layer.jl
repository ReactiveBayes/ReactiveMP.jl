# The input layer.

"""
    InputLayer(dim::Int)

The first element of a layer tuple given to [`FlowModel`](@ref)`(layers)`, which gives the
model's dimension `dim`; `FlowModel((InputLayer(dim), layers...))` is `FlowModel(dim, layers)`.

# Throws

An `AssertionError` unless `dim > 1`.
"""
struct InputLayer <: AbstractLayerPlaceholder
    dim::Int
    function InputLayer(dim::Int)
        @assert dim > 1 "The specified input dimension should be larger than 1."
        return new(dim)
    end
end

getdim(layer::InputLayer) = layer.dim
