export FlowModel, CompiledFlowModel
export compile, nr_params
export getlayers, getforward, getbackward, getjacobian, getinv_jacobian
export forward_jacobian, backward_inv_jacobian
export forward, forward!, backward, backward!
export jacobian, jacobian!, inv_jacobian, inv_jacobian!
export det_jacobian, absdet_jacobian, logdet_jacobian, logabsdet_jacobian
export detinv_jacobian, absdetinv_jacobian, logdetinv_jacobian, logabsdetinv_jacobian

using TupleTools: flatten

import Base: length, eltype

@doc raw"""
The FlowModel structure is the most generic type of Flow model, in which the layers are not constrained to be of a specific type. The FlowModel structure contains the input dimensionality and a tuple of layers and can be constructed as `FlowModel( dim, (layer1, layer2, ...) )`.

Note: this model can be specialized by constraining the types of layers. This potentially allows for more efficient specialized methods that can deal with specifics of these layers, such as triangular jacobian matrices.
"""
struct FlowModel{N,T<:NTuple{N,AbstractLayer}} <: AbstractFlowModel
    dim     :: Int
    layers  :: T
end

@doc raw"""
Creates a flow model with uninitialized parameters.

Input arguments:
- `dim::Int` - input dimensionality.
- `layers<:NTuple{N,AbstractLayer}` - arbitrarily sized tuple containing abstractlayers.

Return arguments:
- `::Flowmodel` - model containing layers of which are appropriately sized according to the input dimensionality.
"""
function FlowModel(dim::Int, layers::T) where { T <: NTuple{N,AbstractLayerPlaceholder} where { N } }
    return FlowModel(dim, flatten(prepare.(dim, layers)))
end
function FlowModel(layers::T) where { T <: NTuple{N, Union{AbstractLayer, AbstractLayerPlaceholder}} where { N } }
    @assert typeof(first(layers)) <: InputLayer "The FlowModel requires an input dimension to be specified. This can be achieved, either by preceding the layers tuple with an integer as `FlowModel(dim, layers)`, or by starting the tuple of layers with an `InputLayer(dim)` as `FlowModel((InputLayer(dim), layers...))`."
    return FlowModel(getdim(first(layers)), flatten(prepare.(getdim(first(layers)), Base.tail(layers))))
end

# prepare function for setting correct sizes in the layers (without assigning the parameters yet!)
prepare(dim::Int, layers::T) where { T <: NTuple{N,Union{AbstractLayer, AbstractLayerPlaceholder}} where { N }} = _prepare(dim, layers)
Broadcast.broadcasted(::typeof(prepare), dim::Int, layers::T) where { T <: NTuple{N,Union{AbstractLayer, AbstractLayerPlaceholder}} where { N }} = broadcast(_prepare, Ref(dim), layers)

@doc raw"""
The CompiledFlowModel structure is the most generic type of compiled Flow model, in which the layers are not constrained to be of a specific type. The FlowModel structure contains the input dimension and a tuple of compiled layers. Do not manually create a CompiledFlowModel! Instead create a FlowModel first and compile it with `compile(model::FlowModel)`. This will make sure that all layers/mappings are configured with the proper dimensionality and with randomly sampled parameters. Alternatively, if you would like to pass your own parameters, call `compile(model::FlowModel, params::Vector)`.

Note: this model can be specialized by constraining the types of layers. This potentially allows for more efficient specialized methods that can deal with specifics of these layers, such as triangular jacobian matrices.
"""
struct CompiledFlowModel{N,T<:NTuple{N,AbstractLayer}} <: AbstractCompiledFlowModel
    dim     :: Int
    layers  :: T
end

@doc raw"""
`compile()` compiles a model by setting its parameters. It randomly sets parameter values in the layers and flows such that inference in the model can be obtained.

Input arguments
- `model::FlowModel` - a model of which the dimensionality of its layers/flows has been initialized, but its parameters have not been set.

Return arguments
- `::CompiledFlowModel` - a compiled model with set parameters, such that it can be used for processing data.
"""
function compile(model::FlowModel)
    # do not create parameters here for decentralized initializers (TODO)
    return CompiledFlowModel(model.dim, compile.(model.layers))
end

@doc raw"""
`compile(model::FlowModel, params::Vector)` lets you initialize a model `model` with a vector of parameters `params`.

Input arguments
- `model::FlowModel` - a model of which the dimensionality of its layers/flows has been initialized, but its parameters have not been set.
- `params::Vector`   - a vector of parameters with which the model should be compiled.

Return arguments
- `::CompiledFlowModel` - a compiled model with set parameters, such that it can be used for processing data.
"""
function compile(model::FlowModel, params::Vector)
    
    # assert whether the parameter vector complies with the model
    @assert nr_params(model) == length(params) "The number of parameters in the model does not match the passed number of parameters."
    
    # compile layers
    return CompiledFlowModel(getdim(model), compile(getlayers(model), params))

end
function compile(x::Tuple, params::Vector)

    # fetch number of parameters in layer
    params_in_layer = nr_params(first(x))

    if params_in_layer > 0
        # if layer has parameters, compile that layer and the other layers with parameters
        return (compile(first(x), params[1:params_in_layer]), compile(Base.tail(x), params[params_in_layer+1:end])...)
    else
        # if layer has no parameters, compile that layer on its own and the other layers with parameters
        return (compile(first(x)), compile(Base.tail(x), params)...)
    end
    
end
function compile(x::Tuple)
    return (compile(first(x)), compile(Base.tail(x))...)
end
# compile functions to stop the recursion
compile(x::Tuple{}, params::Vector) = ()
compile(x::Tuple{})         = ()

# get-functions for the FlowModel structure
getdim(model::FlowModel)                    = model.dim
getdim(model::CompiledFlowModel)            = model.dim
getlayers(model::FlowModel)                 = model.layers
getlayers(model::CompiledFlowModel)         = model.layers
getforward(model::CompiledFlowModel)        = (x) -> forward(model, x)
getbackward(model::CompiledFlowModel)       = (x) -> backward(model, x)
getjacobian(model::CompiledFlowModel)       = (x) -> jacobian(model, x)
getinv_jacobian(model::CompiledFlowModel)   = (x) -> inv_jacobian(model, x)

# custom Base function for the FlowModel structure
eltype(model::CompiledFlowModel) = promote_type(map(eltype, model.layers)...)
length(model::CompiledFlowModel) = length(model.layers)

# fetch number of parameters of model
nr_params(model::FlowModel)         = nr_params(getlayers(model))
nr_params(model::CompiledFlowModel) = nr_params(getlayers(model))
nr_params(layers::T) where { T <: NTuple{N,AbstractLayer} where N} = nr_params(first(layers)) + nr_params(Base.tail(layers))
nr_params(layers::Tuple{}) = return 0



# forward pass through the Flow model
function _forward(model::CompiledFlowModel, input::AbstractVector{T}) where { T <: Real }

    # promote type for allocating output
    Ti = promote_type(eltype(model), T)

    # allocate space for result
    output = zeros(Ti, size(input))

    # perform calculations 
    forward!(output, model, input)

    # return result
    return output
    
end

# when calling forward, redirect to _forward
forward(model::CompiledFlowModel, input::AbstractVector{ <: Real }) = _forward(model, input)

# for broadcasting over forward, fix the model for multiple inputs
Broadcast.broadcasted(::typeof(forward), model::CompiledFlowModel, input::AbstractVector{ <: AbstractVector{ <: Real } }) = broadcast(_forward, Ref(model), input)

# inplace forward pass through the Flow model
function forward!(output::AbstractVector{ <: Real }, model::CompiledFlowModel, input::AbstractVector{T}) where { T <: Real }

    # fetch layers
    layers = getlayers(model)

    # promote type for allocating output
    Ti = promote_type(eltype(model), T)

    # decouple changing input from actual input
    input_new = zeros(Ti, size(input))
    input_new .= input

    # perform forward pass over all layers
    forward!(output, layers, input_new)
    
end

# inplace forward pass through a tuple of layers
function forward!(output::AbstractVector{ <: Real }, layers::T, input::AbstractVector{ <: Real }) where { T <: NTuple{N,AbstractLayer} where N}

    # perform pass over first layer
    forward!(output, first(layers), input)

    # update intermediate input
    input .= output

    # perform pass over all remaining layers
    forward!(output, Base.tail(layers), input)

end

# when no layers are left, stop the inplace recursion
forward!(output::AbstractVector{ <: Real }, layers::Tuple{}, input::AbstractVector{ <: Real }) = return


# backward pass through the Flow model
function _backward(model::CompiledFlowModel, output::AbstractVector{T}) where { T <: Real }

    # promote type for allocating output
    Ti = promote_type(eltype(model), T)

    # allocate space for result
    input = zeros(Ti, size(output))

    # perform calculations 
    backward!(input, model, output)

    # return result
    return input
    
end

# when calling backward, redirect to _backward
backward(model::CompiledFlowModel, output::AbstractVector{ <: Real }) = _backward(model, output)

# for broadcasting over backward, fix the model for multiple inputs
Broadcast.broadcasted(::typeof(backward), model::CompiledFlowModel, output::AbstractVector{ <: AbstractVector{ <: Real } }) = broadcast(_backward, Ref(model), output)

# inplace backward pass through the Flow model
function backward!(input::AbstractVector{ <: Real }, model::CompiledFlowModel, output::AbstractVector{T}) where { T <: Real }

    # fetch layers
    layers = getlayers(model)

    # promote type for allocating output
    Ti = promote_type(eltype(model), T)

    # decouple changing output from actual output
    output_new = zeros(Ti, size(output))
    output_new .= output

    # perform backward pass over all layers
    backward!(input, layers, output_new)
    
end

# inplace backward pass through a tuple of layers
function backward!(input::AbstractVector{ <: Real }, layers::T, output::AbstractVector{ <: Real }) where { T <: NTuple{N,AbstractLayer} where N}

    # perform pass over last layer
    backward!(input, last(layers), output)

    # update intermediate output
    output .= input

    # perform pass over all remaining layers
    backward!(input, Base.front(layers), output)

end

# when no layers are left, stop the inplace recursion
backward!(input::AbstractVector{ <: Real }, layers::Tuple{}, output::AbstractVector{ <: Real }) = return


# joint forward-jacobian of the Flow model
function _forward_jacobian(model::CompiledFlowModel, input::AbstractVector{T}) where { T <: Real }

    # fetch layers
    dim    = getdim(model)
    
    # promote type for allocating output
    Ti = promote_type(eltype(model), T)

    # allocate space for output and jacobian
    output = zeros(Ti, dim)
    J = zeros(Ti, dim, dim)
    for k = 1:dim
        J[k,k] = one(Ti)
    end

    # calculate jacobian
    forward_jacobian!(output, J, model, input)

    # return result
    return output, J

end

# when calling forward_jacobian, redirect to _forward_jacobian
forward_jacobian(model::CompiledFlowModel, input::AbstractVector{ <: Real }) = _forward_jacobian(model, input)

# for broadcasting over forward_jacobian, fix the model for multiple inputs
Broadcast.broadcasted(::typeof(forward_jacobian), model::CompiledFlowModel, input::AbstractVector{ <: AbstractVector{ <: Real } }) = broadcast(_jacobian, Ref(model), input)

# inplace forward_jacobian of the Flow model
function forward_jacobian!(output::AbstractVector{ <: Real }, J::AbstractMatrix{T}, model::CompiledFlowModel, input::AbstractVector{ <: Real }) where { T <: Real }

    # fetch layers
    layers = getlayers(model)
    dim = getdim(model)
    
    # promote type for allocating output
    Ti = promote_type(eltype(model), T)

    # allocate space for intermediate output and jacobian
    input_new = zeros(Ti, dim)
    input_new .= input
    J_new = zeros(Ti, dim, dim)
    J_old = zeros(Ti, dim, dim)
    for k = 1:dim
        J_old[k,k] = one(Ti)
    end

    # calculate forward_jacobian over all layers
    forward_jacobian!(J, J_new, J_old, output, input_new, layers)

end

# inplace forward_jacobian calculation for a tuple of layers
function forward_jacobian!(J::AbstractMatrix{ <: Real }, J_new::AbstractMatrix{ <: Real }, J_old::AbstractMatrix{ <: Real }, output::AbstractVector{ <: Real }, input_new::AbstractVector{ <: Real }, layers::T) where { T <: NTuple{N,AbstractLayer} where N}

    # perform pass through first layers
    forward_jacobian!(J, J_new, J_old, output, input_new, first(layers))

    # update unless we are on the last layer
    if length(layers) > 1
        
        # update intermediate input
        input_new .= output

        # update old jacobian
        J_old .= J

    end

    # calculate forward_jacobian of remaining layers
    forward_jacobian!(J, J_new, J_old, output, input_new, Base.tail(layers))

end

# specialized methods
function forward_jacobian!(J::AbstractMatrix{ <: Real }, J_new::AbstractMatrix{ <: Real }, J_old::AbstractMatrix{ <: Real }, output::AbstractVector{ <: Real }, input_new::AbstractVector{ <: Real }, layer::PermutationLayer)
    # calculate new output
    forward!(output, layer, input_new)
            
    # calculate total jacobian
    mul!(J, jacobian(layer, input_new), J_old)
end

# standard method
function forward_jacobian!(J::AbstractMatrix{ <: Real }, J_new::AbstractMatrix{ <: Real }, J_old::AbstractMatrix{ <: Real }, output::AbstractVector{ <: Real }, input_new::AbstractVector{ <: Real }, layer::AbstractLayer)
    # calculate new jacobian
    forward_jacobian!(output, J_new, first(layers), input_new)

    # calculate total jacobian
    mul!(J, J_new, J_old)
end

# when no layers are left, stop the inplace recursion
forward_jacobian!(J::AbstractMatrix{ <: Real }, J_new::AbstractMatrix{ <: Real }, J_old::AbstractMatrix{ <: Real }, output::AbstractVector{ <: Real }, input_new::AbstractVector{ <: Real }, layers::Tuple{}) = return


# joing backward inverse jacobian of the Flow model
function _backward_inv_jacobian(model::CompiledFlowModel, output::AbstractVector{T}) where { T <: Real }

    # fetch layers
    dim    = getdim(model)
    
    # promote type for allocating output
    Ti = promote_type(eltype(model), T)

    # allocate space for jacobian
    J = zeros(Ti, dim, dim)
    input = zeros(Ti, dim)
    for k = 1:dim
        J[k,k] = one(Ti)
    end

    # calculate jacobian
    backward_inv_jacobian!(input, J, model, output)

    # return result
    return input, J

end

# when calling backward inverse jacobian, redirect to _backward_inv_jacobian
backward_inv_jacobian(model::CompiledFlowModel, output::AbstractVector{ <: Real}) = _backward_inv_jacobian(model, output)

# for broadcasting over backward inverse jacobian, fix the model for multiple inputs
Broadcast.broadcasted(::typeof(backward_inv_jacobian), model::CompiledFlowModel, output::AbstractVector{ <: AbstractVector{ <: Real } }) = broadcast(_backward_inv_jacobian, Ref(model), output)

# inplace inverse backward jacobian of the Flow model
function backward_inv_jacobian!(input::AbstractVector{ <: Real }, J::AbstractMatrix{T}, model::CompiledFlowModel, output::AbstractVector{ <: Real }) where { T <: Real }

    # fetch layers
    layers = getlayers(model)
    dim = getdim(model)
    
    # promote type for allocating output
    Ti = promote_type(eltype(model), T)

    # allocate space for intermediate output and jacobian
    output_new = zeros(Ti, dim)
    output_new .= output
    J_new = zeros(Ti, dim, dim)
    J_old = zeros(Ti, dim, dim)
    for k = 1:dim
        J_old[k,k] = one(Ti)
    end

    # calculate backward inverse jacobian over all layers
    backward_inv_jacobian!(J, J_new, J_old, input, output_new, layers)

end

# inplace backward inverse jacobian calculation for a tuple of layers
function backward_inv_jacobian!(J::AbstractMatrix{ <: Real }, J_new::AbstractMatrix{ <: Real }, J_old::AbstractMatrix{ <: Real }, input::AbstractVector{ <: Real }, output_new::AbstractVector{ <: Real }, layers::T) where { T <: NTuple{N,AbstractLayer} where N}

    # perform backward pass through layers
    backward_inv_jacobian!(J, J_new, J_old, input, output_new, last(layers))

    # update unless we are on the last/first layer
    if length(layers) > 1

        # update intermediate output
        output_new .= input
        
        # update old jacobian
        J_old .= J

    end

    # calculate backward inv jacobian of remaining layers
    backward_inv_jacobian!(J, J_new, J_old, input, output_new, Base.front(layers))

end

# specialized methods
function backward_inv_jacobian!(J::AbstractMatrix{ <: Real }, J_new::AbstractMatrix{ <: Real }, J_old::AbstractMatrix{ <: Real }, input::AbstractVector{ <: Real }, output_new::AbstractVector{ <: Real }, layer::PermutationLayer)
    # perform forward pass over last layer
    backward!(input, last(layers), output_new)
        
    # calculate total jacobian
    mul!(J, inv_jacobian(last(layers), output_new), J_old)
end

# standard method
function backward_inv_jacobian!(J::AbstractMatrix{ <: Real }, J_new::AbstractMatrix{ <: Real }, J_old::AbstractMatrix{ <: Real }, input::AbstractVector{ <: Real }, output_new::AbstractVector{ <: Real }, layer::AbstractLayer)
    # perform forward pass over last layer
    backward_inv_jacobian!(input, J_new, last(layers), output_new)
        
    # calculate total jacobian
    mul!(J, J_new, J_old)
end

# when no layers are left, stop the inplace recursion
backward_inv_jacobian!(J::AbstractMatrix{ <: Real }, J_new::AbstractMatrix{ <: Real }, J_old::AbstractMatrix{ <: Real }, input::AbstractVector{ <: Real }, output_new::AbstractVector{ <: Real }, layers::Tuple{}) = return


# specify jacobian and inv_jacobian functions based on joint functions
_jacobian(model::CompiledFlowModel, input::AbstractVector{ <: Real })      = forward_jacobian(model, input)[2]
jacobian(model::CompiledFlowModel, input::AbstractVector{ <: Real })       = _jacobian(model, input)
Broadcast.broadcasted(::typeof(jacobian), model::CompiledFlowModel, input::AbstractVector{ <: AbstractVector{ <: Real } }) = broadcast(_jacobian, Ref(model), input)
_inv_jacobian(model::CompiledFlowModel, output::AbstractVector{ <: Real }) = backward_inv_jacobian(model, output)[2]
inv_jacobian(model::CompiledFlowModel, output::AbstractVector{ <: Real })  = _inv_jacobian(model, output)
Broadcast.broadcasted(::typeof(inv_jacobian), model::CompiledFlowModel, output::AbstractVector{ <: AbstractVector{ <: Real } }) = broadcast(_inv_jacobian, Ref(model), output)
function jacobian!(J::AbstractMatrix{T1}, model::CompiledFlowModel, input::AbstractVector{T2}) where { T1 <: Real, T2 <: Real }
    
    # fetch dimension
    dim    = getdim(model)
    
    # promote type for allocating output
    T = promote_type(eltype(model), T1, T2)
    
    # allocate space for output and jacobian
    output = zeros(T, dim)
    J .= zero(T1)
    for k = 1:dim
        J[k,k] = one(T1)
    end

    # calculate jacobian
    forward_jacobian!(output, J, model, input)
    
end
function inv_jacobian!(J::AbstractMatrix{T1}, model::CompiledFlowModel, output::AbstractVector{T2}) where { T1 <: Real, T2 <: Real }
    
    # fetch dimension
    dim    = getdim(model)
    
    # promote type for allocating output
    T = promote_type(eltype(model), T1, T2)
    
    # allocate space for input and jacobian
    input = zeros(T, dim)
    J .= zero(T1)
    for k = 1:dim
        J[k,k] = one(T1)
    end

    # calculate jacobian
    backward_inv_jacobian!(input, J, model, output)
    
end

# fallback joint functions over layers
function forward_jacobian!(output::AbstractVector{ <: Real }, J_new::AbstractMatrix{ <: Real }, layer::AbstractLayer, input::AbstractVector{ <: Real })
    forward!(output, layer, input)
    jacobian!(J_new, layer, input)
end
function backward_inv_jacobian!(input::AbstractVector{ <: Real }, J_new::AbstractMatrix{ <: Real }, layer::AbstractLayer, output::AbstractVector{ <: Real })
    backward!(input, layer, output)
    inv_jacobian!(J_new, layer, output)
end

# extra utility functions
det_jacobian(model::CompiledFlowModel, input::AbstractVector{<: Real})           = det(jacobian(model, input))
absdet_jacobian(model::CompiledFlowModel, input::AbstractVector{<: Real})        = abs(det_jacobian(model, input))
logdet_jacobian(model::CompiledFlowModel, input::AbstractVector{<: Real})        = logdet(jacobian(model, input))
logabsdet_jacobian(model::CompiledFlowModel, input::AbstractVector{<: Real})     = logabsdet(jacobian(model, input))

detinv_jacobian(model::CompiledFlowModel, output::AbstractVector{<: Real})       = det(inv_jacobian(model, output))
absdetinv_jacobian(model::CompiledFlowModel, output::AbstractVector{<: Real})    = abs(detinv_jacobian(model, output))
logdetinv_jacobian(model::CompiledFlowModel, output::AbstractVector{<: Real})    = logdet(inv_jacobian(model, output))
logabsdetinv_jacobian(model::CompiledFlowModel, output::AbstractVector{<: Real}) = logabsdet(inv_jacobian(model, output))


# throw an error when the model has not yet been compiled
_forward(model::FlowModel, input::AbstractVector{ <: Real })                                                        = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
forward(model::FlowModel, input::AbstractVector{ <: Real })                                                         = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
Broadcast.broadcasted(::typeof(forward), model::FlowModel, input::AbstractVector{ <: AbstractVector{ <: Real } })   = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
forward!(output::AbstractVector{ <: Real }, model::FlowModel, input::AbstractVector{ <: Real })                     = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
_backward(model::FlowModel, output::AbstractVector{ <: Real })                                                      = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
backward(model::FlowModel, output::AbstractVector{ <: Real })                                                       = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
Broadcast.broadcasted(::typeof(backward), model::FlowModel, output::AbstractVector{ <: AbstractVector{ <: Real } }) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
backward!(input::AbstractVector{ <: Real }, model::FlowModel, output::AbstractVector{ <: Real })                    = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
jacobian!(J_new::AbstractMatrix{ <: Real }, model::FlowModel, input::AbstractVector{ <: Real })                     = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
jacobian(model::FlowModel, input::AbstractVector{ <: Real })                                                        = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
Broadcast.broadcasted(::typeof(jacobian), model::FlowModel, input::AbstractVector{ <: AbstractVector{ <: Real } })  = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
_jacobian(model::FlowModel, input::AbstractVector{ <: Real })                                                       = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
_inv_jacobian(model::FlowModel, output::AbstractVector{ <: Real })                                                  = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
inv_jacobian!(J_new::AbstractMatrix{ <: Real }, model::FlowModel, output::AbstractVector{ <: Real })                = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))

det_jacobian(model::FlowModel, input::AbstractVector{ <: Real })           = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
absdet_jacobian(model::FlowModel, input::AbstractVector{ <: Real })        = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
logdet_jacobian(model::FlowModel, input::AbstractVector{ <: Real })        = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
logabsdet_jacobian(model::FlowModel, input::AbstractVector{ <: Real })     = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))

detinv_jacobian(model::FlowModel, output::AbstractVector{ <: Real })       = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
absdetinv_jacobian(model::FlowModel, output::AbstractVector{ <: Real })    = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
logdetinv_jacobian(model::FlowModel, output::AbstractVector{ <: Real })    = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
logabsdetinv_jacobian(model::FlowModel, output::AbstractVector{ <: Real }) = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))