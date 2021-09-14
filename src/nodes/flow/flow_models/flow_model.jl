## TODO: create a forward-jacobian joint function that calculates both at the same time

export FlowModel
export getlayers, getforward, getbackward, getjacobian, getinv_jacobian
export forward, forward!, backward, backward!
export jacobian, jacobian!, inv_jacobian, inv_jacobian!
export det_jacobian, absdet_jacobian, logdet_jacobian, logabsdet_jacobian
export detinv_jacobian, absdetinv_jacobian, logdetinv_jacobian, logabsdetinv_jacobian

import Base: length, eltype

@doc raw"""

The FlowModel structure is the most generic type of Flow model, in which the layers are not constrained to be of a specific type. The FlowModel structure contains a tuple of layers and can be constructed as `FlowModel( (layer1, layer2, ...) )`.

Note: this model can be specialized by constraining the types of layers. This potentially allows for more efficient specialized methods that can deal with specifics of these layers, such as triangular jacobian matrices.
"""
struct FlowModel{N,T<:NTuple{N,AbstractCouplingLayer}} <: AbstractFlowModel
    layers  :: T
end

# get-functions for the FlowModel structure
getlayers(model::FlowModel)         = model.layers
getforward(model::FlowModel)        = (x) -> forward(model, x)
getbackward(model::FlowModel)       = (x) -> backward(model, x)
getjacobian(model::FlowModel)       = (x) -> jacobian(model, x)
getinv_jacobian(model::FlowModel)   = (x) -> inv_jacobian(model, x)

# custom Base function for the FlowModel structure
eltype(model::FlowModel) = promote_type(map(eltype, model.layers)...)
length(model::FlowModel) = length(model.layers)

# forward pass through the Flow model
function _forward(model::FlowModel, input::Array{T1,1}) where { T1 <: Real }

    # fetch layers
    layers = getlayers(model)

    # promote type for allocating output
    T = promote_type(eltype(model), T1)

    # allocate space for result
    input_new = zeros(T, size(input))
    input_new .= input
    output = zeros(T, size(input))
    
    # pass result along the graph
    for k = 1:length(layers)
        forward!(output, layers[k], input_new)
        if k < length(layers)
            input_new .= output
        end
    end

    # return result
    return output
    
end
forward(model::FlowModel, input::Array{T,1}) where { T <: Real } = _forward(model, input)
Broadcast.broadcasted(::typeof(forward), model::FlowModel, input::Array{Array{T,1},1}) where { T <: Real } = broadcast(_forward, Ref(model), input)

# inplace forward pass through the Flow model
function forward!(output::Array{T1,1}, model::FlowModel, input::Array{T2,1}) where { T1 <: Real, T2 <: Real }

    # fetch layers
    layers = getlayers(model)

    # promote type for allocating output
    T = promote_type(eltype(model), T2)

    # allocate space for result
    input_new = zeros(T, size(input))
    input_new .= input
    
    # pass result along the graph
    for k = 1:length(layers)
        forward!(output, layers[k], input_new)
        if k < length(layers)
            input_new .= output
        end
    end
    
end

# backward pass through the Flow model
function _backward(model::FlowModel, output::Array{T1,1}) where { T1 <: Real }

    # fetch layers
    layers = getlayers(model)

    # promote type for allocating output
    T = promote_type(eltype(model), T1)

    # allocate space for result
    output_new = zeros(T, size(output))
    output_new .= output
    input = zeros(T, size(output))
    
    # pass result along the graph
    for k = length(layers):-1:1
        backward!(input, layers[k], output_new)
        if k > 1
            output_new .= input
        end
    end

    # return result
    return input
    
end
backward(model::FlowModel, output::Array{T,1}) where { T <: Real } = _backward(model, output)
Broadcast.broadcasted(::typeof(backward), model::FlowModel, output::Array{Array{T,1},1}) where { T <: Real } = broadcast(_backward, Ref(model), output)

# inplace backward pass through the Flow model
function backward!(input::Array{T1,1}, model::FlowModel, output::Array{T2,1}) where { T1 <: Real, T2 <: Real }

    # fetch layers
    layers = getlayers(model)

    # promote type for allocating output
    T = promote_type(eltype(model), T2)

    # allocate space for result
    output_new = zeros(T, size(output))
    output_new .= output
    
    # pass result along the graph
    for k = length(layers):-1:1
        backward!(input, layers[k], output_new)
        if k > 1
            output_new .= input
        end
    end
    
end

# jacobian of the Flow model
function _jacobian(model::FlowModel, input::Array{T1,1}) where { T1 <: Real }

    # fetch layers
    layers = getlayers(model)
    
    # promote type for allocating output
    T = promote_type(eltype(model), T1)

    # allocate space for output
    input_new = zeros(T, size(input))
    input_new .= input
    output = zeros(T, size(input))
    J = zeros(T, 2, 2)
    J[1,1] = 1.0
    J[2,2] = 1.0
    J_new = copy(J)

    # pass result along the graph
    for k = 1:length(layers)
        
        # calculate jacobian
        mul!(J_new, jacobian(layers[k], input_new), J)

        # perform forward pass and update inputs
        if k < length(layers)
            forward!(output, layers[k], input_new)
            input_new .= output
            J .= J_new
        end

    end

    # return result
    return J_new

end
jacobian(model::FlowModel, input::Array{T,1}) where { T <: Real } = _jacobian(model, input)
Broadcast.broadcasted(::typeof(jacobian), model::FlowModel, input::Array{Array{T,1},1}) where { T <: Real } = broadcast(_jacobian, Ref(model), input)

# inplace jacobian of the Flow model
function jacobian!(J_new::Array{T1,2}, model::FlowModel, input::Array{T2,1}) where { T1 <: Real, T2 <: Real }

    # fetch layers
    layers = getlayers(model)
    
    # promote type for allocating output
    T = promote_type(eltype(model), T2)

    # allocate space for output
    input_new = zeros(T, size(input))
    input_new .= input
    output = zeros(T, size(input))
    J = zeros(T, 2, 2)
    J[1,1] = 1.0
    J[2,2] = 1.0
    J_new .= J

    # pass result along the graph
    for k = 1:length(layers)
        
        # calculate jacobian
        mul!(J_new, jacobian(layers[k], input_new), J)

        # perform forward pass and update inputs
        if k < length(layers)
            forward!(output, layers[k], input_new)
            input_new .= output
            J .= J_new
        end

    end

end

# inverse jacobian of the Flow model
function _inv_jacobian(model::FlowModel, output::Array{T1,1}) where { T1 <: Real }

    # fetch layers
    layers = getlayers(model)

    # promote type for allocating output
    T = promote_type(eltype(model), T1)

    # allocate space for output
    output_new = zeros(T, size(output))
    output_new .= output
    input = zeros(T, size(output))
    J = zeros(T, 2, 2)
    J[1,1] = 1.0
    J[2,2] = 1.0
    J_new = copy(J)
    
    # pass result along the graph
    for k = length(layers):-1:1

        # calculate jacobian
        mul!(J_new, inv_jacobian(layers[k], output_new), J)

        # perform backward pass and update outputs
        if k > 1
            backward!(input, layers[k], output_new)
            output_new .= input
            J .= J_new
        end

    end

    # return result
    return J_new
    
end
inv_jacobian(model::FlowModel, output::Array{T,1}) where { T <: Real } = _inv_jacobian(model, output)
Broadcast.broadcasted(::typeof(inv_jacobian), model::FlowModel, output::Array{Array{T,1},1}) where { T <: Real } = broadcast(_inv_jacobian, Ref(model), output)

# inplace inverse jacobian of the Flow model
function inv_jacobian!(J_new::Array{T1,2}, model::FlowModel, output::Array{T2,1}) where { T1 <: Real, T2 <: Real }

    # fetch layers
    layers = getlayers(model)

    # promote type for allocating output
    T = promote_type(eltype(model), T2)

    # allocate space for output
    output_new = zeros(T, size(output))
    output_new .= output
    input = zeros(T, size(output))
    J = zeros(T, 2, 2)
    J[1,1] = 1.0
    J[2,2] = 1.0
    J_new .= J
    
    # pass result along the graph
    for k = length(layers):-1:1

        # calculate jacobian
        mul!(J_new, inv_jacobian(layers[k], output_new), J)

        # perform backward pass and update outputs
        if k > 1
            backward!(input, layers[k], output_new)
            output_new .= input
            J .= J_new
        end

    end
    
end

# extra utility functions
det_jacobian(model::FlowModel, input::Array{T,1})           where { T <: Real} = det(jacobian(model, input))
absdet_jacobian(model::FlowModel, input::Array{T,1})        where { T <: Real} = abs(det_jacobian(model, input))
logdet_jacobian(model::FlowModel, input::Array{T,1})        where { T <: Real} = logdet(jacobian(model, input))
logabsdet_jacobian(model::FlowModel, input::Array{T,1})     where { T <: Real} = logabsdet(jacobian(model, input))

detinv_jacobian(model::FlowModel, output::Array{T,1})       where { T <: Real} = det(inv_jacobian(model, output))
absdetinv_jacobian(model::FlowModel, output::Array{T,1})    where { T <: Real} = abs(detinv_jacobian(model, output))
logdetinv_jacobian(model::FlowModel, output::Array{T,1})    where { T <: Real} = logdet(inv_jacobian(model, output))
logabsdetinv_jacobian(model::FlowModel, output::Array{T,1}) where { T <: Real} = logabsdet(inv_jacobian(model, output))