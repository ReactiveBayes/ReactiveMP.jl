export Flow, FlowMeta
export getmodel

# specify Flow factor node
struct Flow end
@node Flow Deterministic [ out, in ]

# specify abstract types for Flow building blocks
abstract type AbstractFlowModel end
abstract type AbstractFlowLayer end
abstract type AbstractNeuralNetwork end

@doc raw"""
The `FlowMeta` structure contains the meta data of the `Flow` factor node. More specifically, it contains the model of the `Flow` factor node. The `FlowMeta` structure can be constructed as `FlowMeta(model)`.

The `FlowMeta` structure is required for the `Flow` factor node and can be included with the Flow node as: ```
    y ~ Flow(x) where { meta = FlowMeta(...) }
```
"""
struct FlowMeta{T <: AbstractFlowModel}
    model   :: T
end
default_meta(::Type{ Flow }) = error("The Flow node requires the meta flag to be explicitly specified. Please create a `FlowMeta` structure for this purpose and include it with the Flow node as: `y ~ Flow(x) where { meta = FlowMeta(...) }` ")

# get-functions for the FlowMeta structure
getmodel(meta::FlowMeta) = meta.model

# include neural networks
include("neural_networks/planar_map.jl")

# include layers
include("flow_layers/nice_layer.jl")
include("flow_layers/reverse_nice_layer.jl")

# include models
include("flow_models/flow_model.jl")