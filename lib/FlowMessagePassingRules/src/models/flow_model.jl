# The flow model and its compiled form.

@doc raw"""
    FlowModel([rng,] dim::Int, layers::Tuple)
    FlowModel([rng,] layers::Tuple)

The FlowModel structure is the most generic type of Flow model, in which the layers are not constrained to be of a specific type. The FlowModel structure contains the input dimensionality and a tuple of layers and can be constructed as `FlowModel( dim, (layer1, layer2, ...) )`, or as `FlowModel( (InputLayer(dim), layer1, layer2, ...) )`.

It creates a flow model with uninitialized parameters, with its layers appropriately sized
according to the input dimensionality `dim`. Its random permutations, those of a
[`PermutationLayer`](@ref)`()` and of an [`AdditiveCouplingLayer`](@ref) with `permute = true`,
are drawn from `rng`, by default the task's generator.

Input arguments:
- `rng::AbstractRNG` - the generator the random permutations are drawn from, optional.
- `dim::Int` - input dimensionality.
- `layers<:NTuple{N,AbstractLayer}` - arbitrarily sized tuple containing abstractlayers.

Return arguments:
- `::FlowModel` - model containing layers of which are appropriately sized according to the input dimensionality.

Note: this model can be specialized by constraining the types of layers. This potentially allows for more efficient specialized methods that can deal with specifics of these layers, such as triangular jacobian matrices.
"""
struct FlowModel{N, T <: NTuple{N, AbstractLayer}} <: AbstractFlowModel
    dim::Int
    layers::T
end

function FlowModel(
        dim::Int, layers::T
    ) where {T <: NTuple{N, AbstractLayerPlaceholder} where {N}}
    return FlowModel(default_rng(), dim, layers)
end
# An empty tuple is both a tuple of layers and a tuple of placeholders; this
# method resolves it: a model without layers.
FlowModel(dim::Int, layers::Tuple{}) = FlowModel{0, Tuple{}}(dim, layers)
function FlowModel(
        rng::AbstractRNG, dim::Int, layers::T
    ) where {T <: NTuple{N, AbstractLayerPlaceholder} where {N}}
    return FlowModel(dim, flatten(prepare(rng, dim, layers)))
end
function FlowModel(
        layers::T
    ) where {
        T <: NTuple{N, Union{AbstractLayer, AbstractLayerPlaceholder}} where {N},
    }
    return FlowModel(default_rng(), layers)
end
function FlowModel(
        rng::AbstractRNG, layers::T
    ) where {
        T <: NTuple{N, Union{AbstractLayer, AbstractLayerPlaceholder}} where {N},
    }
    @assert typeof(first(layers)) <: InputLayer "The FlowModel requires an input dimension to be specified. This can be achieved, either by preceding the layers tuple with an integer as `FlowModel(dim, layers)`, or by starting the tuple of layers with an `InputLayer(dim)` as `FlowModel((InputLayer(dim), layers...))`."
    return FlowModel(
        getdim(first(layers)),
        flatten(prepare(rng, getdim(first(layers)), Base.tail(layers))),
    )
end

# prepare function for setting correct sizes in the layers (without assigning the parameters yet!),
# in order, each layer drawing from `rng`
prepare(
    rng::AbstractRNG, dim::Int, layers::T
) where {
    T <: NTuple{N, Union{AbstractLayer, AbstractLayerPlaceholder}} where {N},
} = map(layer -> _prepare(rng, dim, layer), layers)

@doc raw"""
    CompiledFlowModel

The CompiledFlowModel structure is the most generic type of compiled Flow model, in which the layers are not constrained to be of a specific type. The FlowModel structure contains the input dimension and a tuple of compiled layers. Do not manually create a CompiledFlowModel! Instead create a FlowModel first and compile it with `compile(model::FlowModel)`. This will make sure that all layers/mappings are configured with the proper dimensionality and with randomly sampled parameters. Alternatively, if you would like to pass your own parameters, call `compile(model::FlowModel, params::Vector)`.

Its output, Jacobians and inverse are [`FlowMessagePassingRules.forward`](@ref),
[`FlowMessagePassingRules.backward`](@ref), [`FlowMessagePassingRules.jacobian`](@ref),
[`FlowMessagePassingRules.inv_jacobian`](@ref), [`FlowMessagePassingRules.forward_jacobian`](@ref)
and [`FlowMessagePassingRules.backward_inv_jacobian`](@ref), and `eltype(model)` is the element type
of its parameters.

Note: this model can be specialized by constraining the types of layers. This potentially allows for more efficient specialized methods that can deal with specifics of these layers, such as triangular jacobian matrices.
"""
struct CompiledFlowModel{N, T <: NTuple{N, AbstractLayer}} <:
    AbstractCompiledFlowModel
    dim::Int
    layers::T
end

@doc raw"""
    compile([rng,] model::FlowModel)

`compile()` compiles a model by setting its parameters. It randomly sets parameter values in the layers and flows such that inference in the model can be obtained, drawing them from `rng`, by default the task's generator.

Input arguments
- `rng::AbstractRNG` - the generator the parameters are drawn from, optional.
- `model::FlowModel` - a model of which the dimensionality of its layers/flows has been initialized, but its parameters have not been set.

Return arguments
- `::CompiledFlowModel` - a compiled model with set parameters, such that it can be used for processing data.
"""
compile(model::FlowModel) = compile(default_rng(), model)
function compile(rng::AbstractRNG, model::FlowModel)
    # do not create parameters here for decentralized initializers (TODO)
    return CompiledFlowModel(model.dim, compile(rng, model.layers))
end

@doc raw"""
    compile(model::FlowModel, params::Vector)

`compile(model::FlowModel, params::Vector)` lets you initialize a model `model` with a vector of parameters `params`.

Input arguments
- `model::FlowModel` - a model of which the dimensionality of its layers/flows has been initialized, but its parameters have not been set.
- `params::Vector`   - a vector of parameters with which the model should be compiled.

Return arguments
- `::CompiledFlowModel` - a compiled model with set parameters, such that it can be used for processing data.
"""
function compile(model::FlowModel, params::Vector)

    @assert nr_params(model) == length(params) "The number of parameters in the model does not match the passed number of parameters."

    return CompiledFlowModel(getdim(model), compile(getlayers(model), params))
end
function compile(x::Tuple, params::Vector)

    params_in_layer = nr_params(first(x))

    if params_in_layer > 0
        # if layer has parameters, compile that layer and the other layers with parameters
        return (
            compile(first(x), params[1:params_in_layer]),
            compile(Base.tail(x), params[(params_in_layer + 1):end])...,
        )
    else
        # if layer has no parameters, compile that layer on its own and the other layers with parameters
        return (compile(first(x)), compile(Base.tail(x), params)...)
    end
end
compile(x::Tuple) = compile(default_rng(), x)
function compile(rng::AbstractRNG, x::Tuple)
    first_compiled = compile(rng, first(x))
    return (first_compiled, compile(rng, Base.tail(x))...)
end
# compile functions to stop the recursion
compile(x::Tuple{}, params::Vector) = ()
compile(x::Tuple{}) = ()
compile(rng::AbstractRNG, x::Tuple{}) = ()

"""
    getlayers(model)

The tuple of layers of a [`FlowModel`](@ref) or of a [`CompiledFlowModel`](@ref).
"""
function getlayers end

# get-functions for the FlowModel structure
getdim(model::FlowModel) = model.dim
getdim(model::CompiledFlowModel) = model.dim
getlayers(model::FlowModel) = model.layers
getlayers(model::CompiledFlowModel) = model.layers
getforward(model::CompiledFlowModel) = (x) -> forward(model, x)
getbackward(model::CompiledFlowModel) = (x) -> backward(model, x)
getjacobian(model::CompiledFlowModel) = (x) -> jacobian(model, x)
getinv_jacobian(model::CompiledFlowModel) = (x) -> inv_jacobian(model, x)

"""
    eltype(model::CompiledFlowModel)

The element type of the compiled flow `model`'s parameters, promoted over its layers.
"""
Base.eltype(model::CompiledFlowModel) = promote_type(map(eltype, model.layers)...)
Base.length(model::CompiledFlowModel) = length(model.layers)

"""
    nr_params(model)

The number of parameters of a flow model, of a layer or of a coupling flow, compiled or not: the
length of the vector `compile(model, params)` takes.
"""
function nr_params end

# fetch number of parameters of model
nr_params(model::FlowModel) = nr_params(getlayers(model))
nr_params(model::CompiledFlowModel) = nr_params(getlayers(model))
nr_params(layers::T) where {T <: NTuple{N, AbstractLayer} where {N}} =
    nr_params(first(layers)) + nr_params(Base.tail(layers))
nr_params(layers::Tuple{}) = return 0

# forward pass through the Flow model
function _forward(
        model::CompiledFlowModel, input::AbstractVector{T}
    ) where {T <: Real}

    Ti = promote_type(eltype(model), T)

    output = zeros(Ti, size(input))

    forward!(output, model, input)

    return output
end

# when calling forward, redirect to _forward
forward(model::CompiledFlowModel, input::AbstractVector{<:Real}) =
    _forward(model, input)

# for broadcasting over forward, fix the model for multiple inputs
Broadcast.broadcasted(
    ::typeof(forward),
    model::CompiledFlowModel,
    input::AbstractVector{<:AbstractVector{<:Real}},
) = broadcast(_forward, Ref(model), input)

# inplace forward pass through the Flow model
function forward!(
        output::AbstractVector{<:Real},
        model::CompiledFlowModel,
        input::AbstractVector{T},
    ) where {T <: Real}

    layers = getlayers(model)

    Ti = promote_type(eltype(model), T)

    # decouple changing input from actual input
    input_new = zeros(Ti, size(input))
    input_new .= input

    return forward!(output, layers, input_new)
end

# inplace forward pass through a tuple of layers
function forward!(
        output::AbstractVector{<:Real}, layers::T, input::AbstractVector{<:Real}
    ) where {T <: NTuple{N, AbstractLayer} where {N}}

    forward!(output, first(layers), input)

    input .= output

    return forward!(output, Base.tail(layers), input)
end

# when no layers are left, stop the inplace recursion
forward!(
    output::AbstractVector{<:Real},
    layers::Tuple{},
    input::AbstractVector{<:Real},
) = return nothing

# backward pass through the Flow model
function _backward(
        model::CompiledFlowModel, output::AbstractVector{T}
    ) where {T <: Real}

    Ti = promote_type(eltype(model), T)

    input = zeros(Ti, size(output))

    backward!(input, model, output)

    return input
end

# when calling backward, redirect to _backward
backward(model::CompiledFlowModel, output::AbstractVector{<:Real}) =
    _backward(model, output)

# for broadcasting over backward, fix the model for multiple inputs
Broadcast.broadcasted(
    ::typeof(backward),
    model::CompiledFlowModel,
    output::AbstractVector{<:AbstractVector{<:Real}},
) = broadcast(_backward, Ref(model), output)

# inplace backward pass through the Flow model
function backward!(
        input::AbstractVector{<:Real},
        model::CompiledFlowModel,
        output::AbstractVector{T},
    ) where {T <: Real}

    layers = getlayers(model)

    Ti = promote_type(eltype(model), T)

    # decouple changing output from actual output
    output_new = zeros(Ti, size(output))
    output_new .= output

    return backward!(input, layers, output_new)
end

# inplace backward pass through a tuple of layers
function backward!(
        input::AbstractVector{<:Real}, layers::T, output::AbstractVector{<:Real}
    ) where {T <: NTuple{N, AbstractLayer} where {N}}

    backward!(input, last(layers), output)

    output .= input

    return backward!(input, Base.front(layers), output)
end

# when no layers are left, stop the inplace recursion
backward!(
    input::AbstractVector{<:Real},
    layers::Tuple{},
    output::AbstractVector{<:Real},
) = return nothing

# joint forward-jacobian of the Flow model
function _forward_jacobian(
        model::CompiledFlowModel, input::AbstractVector{T}
    ) where {T <: Real}

    dim = getdim(model)

    Ti = promote_type(eltype(model), T)

    output = zeros(Ti, dim)
    J = zeros(Ti, dim, dim)
    for k in 1:dim
        J[k, k] = one(Ti)
    end

    forward_jacobian!(output, J, model, input)

    return output, J
end

# when calling forward_jacobian, redirect to _forward_jacobian
forward_jacobian(model::CompiledFlowModel, input::AbstractVector{<:Real}) =
    _forward_jacobian(model, input)

# for broadcasting over forward_jacobian, fix the model for multiple inputs
Broadcast.broadcasted(
    ::typeof(forward_jacobian),
    model::CompiledFlowModel,
    input::AbstractVector{<:AbstractVector{<:Real}},
) = broadcast(_jacobian, Ref(model), input)

# inplace forward_jacobian of the Flow model
function forward_jacobian!(
        output::AbstractVector{<:Real},
        J::AbstractMatrix{T},
        model::CompiledFlowModel,
        input::AbstractVector{<:Real},
    ) where {T <: Real}

    layers = getlayers(model)
    dim = getdim(model)

    Ti = promote_type(eltype(model), T)

    input_new = zeros(Ti, dim)
    input_new .= input
    J_new = zeros(Ti, dim, dim)
    J_old = zeros(Ti, dim, dim)
    for k in 1:dim
        J_old[k, k] = one(Ti)
    end

    return forward_jacobian!(J, J_new, J_old, output, input_new, layers)
end

# inplace forward_jacobian calculation for a tuple of layers
function forward_jacobian!(
        J::AbstractMatrix{<:Real},
        J_new::AbstractMatrix{<:Real},
        J_old::AbstractMatrix{<:Real},
        output::AbstractVector{<:Real},
        input_new::AbstractVector{<:Real},
        layers::T,
    ) where {T <: NTuple{N, AbstractLayer} where {N}}

    forward_jacobian!(J, J_new, J_old, output, input_new, first(layers))

    # update unless we are on the last layer
    if length(layers) > 1

        input_new .= output

        J_old .= J
    end

    return forward_jacobian!(J, J_new, J_old, output, input_new, Base.tail(layers))
end

# specialized methods
function forward_jacobian!(
        J::AbstractMatrix{<:Real},
        J_new::AbstractMatrix{<:Real},
        J_old::AbstractMatrix{<:Real},
        output::AbstractVector{<:Real},
        input_new::AbstractVector{<:Real},
        layer::PermutationLayer,
    )
    forward!(output, layer, input_new)

    return mul!(J, jacobian(layer, input_new), J_old)
end

# standard method
function forward_jacobian!(
        J::AbstractMatrix{<:Real},
        J_new::AbstractMatrix{<:Real},
        J_old::AbstractMatrix{<:Real},
        output::AbstractVector{<:Real},
        input_new::AbstractVector{<:Real},
        layer::AbstractLayer,
    )
    forward_jacobian!(output, J_new, layer, input_new)

    return mul!(J, J_new, J_old)
end

# when no layers are left, stop the inplace recursion
forward_jacobian!(
    J::AbstractMatrix{<:Real},
    J_new::AbstractMatrix{<:Real},
    J_old::AbstractMatrix{<:Real},
    output::AbstractVector{<:Real},
    input_new::AbstractVector{<:Real},
    layers::Tuple{},
) = return nothing

# joing backward inverse jacobian of the Flow model
function _backward_inv_jacobian(
        model::CompiledFlowModel, output::AbstractVector{T}
    ) where {T <: Real}

    dim = getdim(model)

    Ti = promote_type(eltype(model), T)

    J = zeros(Ti, dim, dim)
    input = zeros(Ti, dim)
    for k in 1:dim
        J[k, k] = one(Ti)
    end

    backward_inv_jacobian!(input, J, model, output)

    return input, J
end

# when calling backward inverse jacobian, redirect to _backward_inv_jacobian
backward_inv_jacobian(
    model::CompiledFlowModel, output::AbstractVector{<:Real}
) = _backward_inv_jacobian(model, output)

# for broadcasting over backward inverse jacobian, fix the model for multiple inputs
Broadcast.broadcasted(
    ::typeof(backward_inv_jacobian),
    model::CompiledFlowModel,
    output::AbstractVector{<:AbstractVector{<:Real}},
) = broadcast(_backward_inv_jacobian, Ref(model), output)

# inplace inverse backward jacobian of the Flow model
function backward_inv_jacobian!(
        input::AbstractVector{<:Real},
        J::AbstractMatrix{T},
        model::CompiledFlowModel,
        output::AbstractVector{<:Real},
    ) where {T <: Real}

    layers = getlayers(model)
    dim = getdim(model)

    Ti = promote_type(eltype(model), T)

    output_new = zeros(Ti, dim)
    output_new .= output
    J_new = zeros(Ti, dim, dim)
    J_old = zeros(Ti, dim, dim)
    for k in 1:dim
        J_old[k, k] = one(Ti)
    end

    return backward_inv_jacobian!(J, J_new, J_old, input, output_new, layers)
end

# inplace backward inverse jacobian calculation for a tuple of layers
function backward_inv_jacobian!(
        J::AbstractMatrix{<:Real},
        J_new::AbstractMatrix{<:Real},
        J_old::AbstractMatrix{<:Real},
        input::AbstractVector{<:Real},
        output_new::AbstractVector{<:Real},
        layers::T,
    ) where {T <: NTuple{N, AbstractLayer} where {N}}

    backward_inv_jacobian!(J, J_new, J_old, input, output_new, last(layers))

    # update unless we are on the last/first layer
    if length(layers) > 1

        output_new .= input

        J_old .= J
    end

    return backward_inv_jacobian!(
        J, J_new, J_old, input, output_new, Base.front(layers)
    )
end

# specialized methods
function backward_inv_jacobian!(
        J::AbstractMatrix{<:Real},
        J_new::AbstractMatrix{<:Real},
        J_old::AbstractMatrix{<:Real},
        input::AbstractVector{<:Real},
        output_new::AbstractVector{<:Real},
        layer::PermutationLayer,
    )
    backward!(input, layer, output_new)

    return mul!(J, inv_jacobian(layer, output_new), J_old)
end

# standard method
function backward_inv_jacobian!(
        J::AbstractMatrix{<:Real},
        J_new::AbstractMatrix{<:Real},
        J_old::AbstractMatrix{<:Real},
        input::AbstractVector{<:Real},
        output_new::AbstractVector{<:Real},
        layer::AbstractLayer,
    )
    backward_inv_jacobian!(input, J_new, layer, output_new)

    return mul!(J, J_new, J_old)
end

# when no layers are left, stop the inplace recursion
backward_inv_jacobian!(
    J::AbstractMatrix{<:Real},
    J_new::AbstractMatrix{<:Real},
    J_old::AbstractMatrix{<:Real},
    input::AbstractVector{<:Real},
    output_new::AbstractVector{<:Real},
    layers::Tuple{},
) = return nothing

# specify jacobian and inv_jacobian functions based on joint functions
_jacobian(model::CompiledFlowModel, input::AbstractVector{<:Real}) =
    forward_jacobian(model, input)[2]
jacobian(model::CompiledFlowModel, input::AbstractVector{<:Real}) =
    _jacobian(model, input)
Broadcast.broadcasted(
    ::typeof(jacobian),
    model::CompiledFlowModel,
    input::AbstractVector{<:AbstractVector{<:Real}},
) = broadcast(_jacobian, Ref(model), input)
_inv_jacobian(model::CompiledFlowModel, output::AbstractVector{<:Real}) =
    backward_inv_jacobian(model, output)[2]
inv_jacobian(model::CompiledFlowModel, output::AbstractVector{<:Real}) =
    _inv_jacobian(model, output)
Broadcast.broadcasted(
    ::typeof(inv_jacobian),
    model::CompiledFlowModel,
    output::AbstractVector{<:AbstractVector{<:Real}},
) = broadcast(_inv_jacobian, Ref(model), output)
function jacobian!(
        J::AbstractMatrix{T1}, model::CompiledFlowModel, input::AbstractVector{T2}
    ) where {T1 <: Real, T2 <: Real}

    dim = getdim(model)

    T = promote_type(eltype(model), T1, T2)

    output = zeros(T, dim)
    J .= zero(T1)
    for k in 1:dim
        J[k, k] = one(T1)
    end

    return forward_jacobian!(output, J, model, input)
end
function inv_jacobian!(
        J::AbstractMatrix{T1}, model::CompiledFlowModel, output::AbstractVector{T2}
    ) where {T1 <: Real, T2 <: Real}

    dim = getdim(model)

    T = promote_type(eltype(model), T1, T2)

    input = zeros(T, dim)
    J .= zero(T1)
    for k in 1:dim
        J[k, k] = one(T1)
    end

    return backward_inv_jacobian!(input, J, model, output)
end

# fallback joint functions over layers
function forward_jacobian!(
        output::AbstractVector{<:Real},
        J_new::AbstractMatrix{<:Real},
        layer::AbstractLayer,
        input::AbstractVector{<:Real},
    )
    forward!(output, layer, input)
    return jacobian!(J_new, layer, input)
end
function backward_inv_jacobian!(
        input::AbstractVector{<:Real},
        J_new::AbstractMatrix{<:Real},
        layer::AbstractLayer,
        output::AbstractVector{<:Real},
    )
    backward!(input, layer, output)
    return inv_jacobian!(J_new, layer, output)
end

# extra utility functions
det_jacobian(model::CompiledFlowModel, input::AbstractVector{<:Real}) = det(jacobian(model, input))
absdet_jacobian(model::CompiledFlowModel, input::AbstractVector{<:Real}) = abs(det_jacobian(model, input))
logdet_jacobian(model::CompiledFlowModel, input::AbstractVector{<:Real}) = logdet(jacobian(model, input))
logabsdet_jacobian(model::CompiledFlowModel, input::AbstractVector{<:Real}) = logabsdet(jacobian(model, input))

detinv_jacobian(model::CompiledFlowModel, output::AbstractVector{<:Real}) = det(inv_jacobian(model, output))
absdetinv_jacobian(model::CompiledFlowModel, output::AbstractVector{<:Real}) = abs(detinv_jacobian(model, output))
logdetinv_jacobian(model::CompiledFlowModel, output::AbstractVector{<:Real}) = logdet(inv_jacobian(model, output))
logabsdetinv_jacobian(model::CompiledFlowModel, output::AbstractVector{<:Real}) = logabsdet(inv_jacobian(model, output))

# throw an error when the model has not yet been compiled
_forward(model::FlowModel, input::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
forward(model::FlowModel, input::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
Broadcast.broadcasted(::typeof(forward), model::FlowModel, input::AbstractVector{<:AbstractVector{<:Real}}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
forward!(output::AbstractVector{<:Real}, model::FlowModel, input::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
_backward(model::FlowModel, output::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
backward(model::FlowModel, output::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
Broadcast.broadcasted(::typeof(backward), model::FlowModel, output::AbstractVector{<:AbstractVector{<:Real}}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
backward!(input::AbstractVector{<:Real}, model::FlowModel, output::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
jacobian!(J_new::AbstractMatrix{<:Real}, model::FlowModel, input::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
jacobian(model::FlowModel, input::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
Broadcast.broadcasted(::typeof(jacobian), model::FlowModel, input::AbstractVector{<:AbstractVector{<:Real}}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
_jacobian(model::FlowModel, input::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
_inv_jacobian(model::FlowModel, output::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
inv_jacobian!(J_new::AbstractMatrix{<:Real}, model::FlowModel, output::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))

det_jacobian(model::FlowModel, input::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
absdet_jacobian(model::FlowModel, input::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
logdet_jacobian(model::FlowModel, input::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
logabsdet_jacobian(model::FlowModel, input::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))

detinv_jacobian(model::FlowModel, output::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
absdetinv_jacobian(model::FlowModel, output::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
logdetinv_jacobian(model::FlowModel, output::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
logabsdetinv_jacobian(model::FlowModel, output::AbstractVector{<:Real}) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
