# The abstract types a flow model is built from, and the functions the Flow node's rules call on a
# compiled model.

abstract type AbstractFlowModel end
abstract type AbstractCompiledFlowModel end
abstract type AbstractLayer end
abstract type AbstractLayerPlaceholder end
abstract type AbstractCouplingLayer <: AbstractLayer end
abstract type AbstractCouplingFlow end
abstract type AbstractCouplingFlowEmpty end
abstract type AbstractCouplingFlowPlaceholder end

public forward, backward, jacobian, inv_jacobian, forward_jacobian, backward_inv_jacobian

"""
    forward(model::CompiledFlowModel, input::AbstractVector{<:Real})

The output of the compiled flow `model` at `input`, passing it through every layer in order. It
has methods for the layers and the coupling flows too, and broadcasts over a vector of inputs
with the model fixed, `forward.(model, inputs)`.
"""
function forward end

"""
    backward(model::CompiledFlowModel, output::AbstractVector{<:Real})

The inverse of [`FlowMessagePassingRules.forward`](@ref): the input of the compiled flow `model`
that gives `output`, passing it back through every layer in reverse order. It broadcasts over a
vector of outputs with the model fixed.
"""
function backward end

"""
    jacobian(model::CompiledFlowModel, input::AbstractVector{<:Real})

The Jacobian matrix of the compiled flow `model` at `input`, the product of its layers' Jacobians.
It has methods for the layers and the coupling flows too, and broadcasts over a vector of inputs
with the model fixed.
"""
function jacobian end

"""
    inv_jacobian(model::CompiledFlowModel, output::AbstractVector{<:Real})

The Jacobian matrix of the inverse flow, [`FlowMessagePassingRules.backward`](@ref), at `output`:
the inverse of `jacobian(model, backward(model, output))`. It broadcasts over a vector of outputs
with the model fixed.
"""
function inv_jacobian end

"""
    forward_jacobian(model::CompiledFlowModel, input::AbstractVector{<:Real})

The output and the Jacobian matrix of the compiled flow `model` at `input` in one pass, the tuple
`(forward(model, input), jacobian(model, input))`.
"""
function forward_jacobian end

"""
    backward_inv_jacobian(model::CompiledFlowModel, output::AbstractVector{<:Real})

The input and the inverse flow's Jacobian matrix at `output` in one pass, the tuple
`(backward(model, output), inv_jacobian(model, output))`.
"""
function backward_inv_jacobian end
