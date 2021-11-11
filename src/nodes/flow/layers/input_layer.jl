export InputLayer

@doc raw"""
The input layer specifies the input dimension to a flow model.

```julia
layer = InputLayer(3)
```
"""
struct InputLayer <: AbstractLayerPlaceholder
    dim :: Int
    function InputLayer(dim::Int)
        @assert dim > 1 "The specified input dimension should be larger than 1."
        return new(dim)
    end
end

getdim(layer::InputLayer) = layer.dim