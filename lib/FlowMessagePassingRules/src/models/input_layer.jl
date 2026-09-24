# The input layer, from v6's `layers/input_layer.jl`.

@doc raw"""
    InputLayer(dim::Int)

The input layer specifies the input dimension to a flow model, `dim > 1`. It starts the tuple of
layers given to [`FlowModel`](@ref), in place of the dimension.

```julia
layer = InputLayer(3)
```
"""
struct InputLayer <: AbstractLayerPlaceholder
    dim::Int
    function InputLayer(dim::Int)
        @assert dim > 1 "The specified input dimension should be larger than 1."
        return new(dim)
    end
end

getdim(layer::InputLayer) = layer.dim
