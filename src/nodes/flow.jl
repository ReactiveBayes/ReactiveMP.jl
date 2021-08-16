export FlowModel, FlowLayer, NeuralNetwork, Parameter, PlanarMap, NiceFlowLayer, NiceFlowModel, FlowMeta

import Base: +, -, *, /, length, iterate
import LinearAlgebra: dot

@node Flow Deterministic [ out, in ]

abstract type FlowModel end
abstract type FlowLayer end
abstract type NeuralNetwork end

mutable struct Parameter{T}
    value   :: T
end

struct PlanarMap{T1} <: NeuralNetwork
    u       :: Parameter{T1}
    w       :: Parameter{T1}
    b       :: Parameter{Float64}
end

struct NiceFlowLayer{T <: NeuralNetwork} <: FlowLayer
    f       :: T
end

struct NiceFlowModel{T <: NiceFlowLayer} <: FlowModel
    layers  :: Array{T, 1}
end

struct FlowMeta
    model   :: FlowModel
end

default_meta(::Type{ Flow }) = error("Flow node requires meta flag to be explicitly specified")

## Parameter methods
+(x::Parameter, y::Parameter)   = x.value + y.value
+(x::Any, y::Parameter)         = x + y.value
+(x::Parameter, y::Any)         = x.value + y

-(x::Parameter, y::Parameter)   = x.value - y.value
-(x::Any, y::Parameter)         = x - y.value
-(x::Parameter, y::Any)         = x.value - y

*(x::Parameter, y::Parameter)   = x.value * y.value
*(x::Any, y::Parameter)         = x * y.value
*(x::Parameter, y::Any)         = x.value * y

/(x::Parameter, y::Parameter)   = x.value / y.value
/(x::Any, y::Parameter)         = x / y.value
/(x::Parameter, y::Any)         = x.value / y

dot(x::Parameter, y::Parameter) = dot(x.value, y.value)
dot(x::Any, y::Parameter)       = dot(x, y.value)
dot(x::Parameter, y::Any)       = dot(x.value, y)

length(x::Parameter)            = length(x.value)

iterate(x::Parameter)           = iterate(x.value)
iterate(x::Parameter, i::Int64) = iterate(x.value, i)

getvalue(x::Parameter)          = x.value

function setvalue!(x::Parameter{T}, value::T) where { T }
    x.value = value
end;


## PlanarMap methods
function PlanarMap(u::T1, w::T1, b::T2) where { T1, T2 <: Real}
    return PlanarMap(Parameter(u), Parameter(w), Parameter(floor(b)))
end
function PlanarMap(dim::Int64)
    return PlanarMap(randn(dim), randn(dim), randn())
end
function PlanarMap()
    return PlanarMap(randn(), randn(), randn())
end

getu(f::PlanarMap)              = f.u
getw(f::PlanarMap)              = f.w
getb(f::PlanarMap)              = f.b
getall(f::PlanarMap)            = f.u, f.w, f.b
getvalues(f::PlanarMap)         = getvalue(f.u), getvalue(f.w), getvalue(f.b)

function setu!(f::PlanarMap{T}, u::T) where { T }
    f.u = u
end

function setw!(f::PlanarMap{T}, w::T) where { T }
    f.w = w
end

function setb!(f::PlanarMap, b::T) where { T <: Real }
    f.b = b
end

function forward(f::PlanarMap{T}, input::T) where { T }

    # fetch values
    u, w, b = getvalues(f)
    
    # calculate result (optimized)
    result = copy(u)
    result .*= tanh(dot(w, input) + b)
    result .+= input

    # return result
    return result

end

function forward(f::PlanarMap{T}, input::T) where { T <: Real }

    # fetch values
    u, w, b = getvalues(f)
    
    # calculate result (optimized)
    result = copy(u)
    result *= tanh(dot(w, input) + b)
    result += input

    # return result
    return result

end

function jacobian(f::PlanarMap{T}, input::T) where { T }

    # fetch values 
    u, w, b = getvalues(f)

    # calculate result (optimized)
    result = u*w'
    result .*= dtanh(dot(w, input) + b)
    @inbounds for k = 1:length(input)
        result[k,k] += 1.0
    end

    # return result
    return result

end

function jacobian(f::PlanarMap{T}, input::T) where { T <: Real }

    # fetch values 
    u, w, b = getvalues(f)

    # calculate result (optimized)
    result = u * w * dtanh(w * input + b) + 1

    # return result
    return result

end

function det_jacobian(f::PlanarMap{T}, input::T) where { T }

    # fetch values
    u, w, b = getvalues(f)

    # return result
    return 1 + dot(u, w)*dtanh(dot(w, input) + b)

end

absdet_jacobian(f::PlanarMap{T}, input::T) where { T } = abs(det_jacobian(f, input))
logabsdet_jacobian(f::PlanarMap{T}, input::T) where { T } = log(absdet_jacobian(f, input))


## NiceFlowLayer methods
getf(layer::NiceFlowLayer)              = layer.f
getmap(layer::NiceFlowLayer)            = layer.f

function forward(layer::NiceFlowLayer, input::Array{Float64,1}) 

    # check dimensionality
    @assert length(input) == 2 "The NiceFlowLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # determine result
    result = [input[1], input[2] + forward(f, input[1])]

    # return result
    return result
    
end

function forward!(output::Array{Float64,1}, layer::NiceFlowLayer, input::Array{Float64,1})

    # check dimensionality
    @assert length(input) == 2 "The NiceFlowLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # determine result
    output[1] = input[1] 
    output[2] = input[2] 
    output[2] += forward(f, input[1])
    
end

function backward(layer::NiceFlowLayer, output::Array{Float64,1})

    # check dimensionality
    @assert length(output) == 2 "The NiceFlowLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # determine result
    result = [output[1], output[2] - forward(f, output[1])]

    # return result
    return result
    
end

function backward!(input::Array{Float64,1}, layer::NiceFlowLayer, output::Array{Float64,1})

    # check dimensionality
    @assert length(output) == 2 "The NiceFlowLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # determine result
    input[1] = output[1]
    input[2] = output[2] - forward(f, output[1])
    
end

function jacobian(layer::NiceFlowLayer, input::Array{Float64,1})

    # check dimensionality
    @assert length(input) == 2 "The NiceFlowLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # determine result  
    result = zeros(Float64, 2,2)
    result[1,1] = 1.0
    result[2,1] = jacobian(f,input[1])
    result[2,2] = 1.0
    
    # return result
    return LowerTriangular(result)
    
end

function inv_jacobian(layer::NiceFlowLayer, output::Array{Float64,1})

    # check dimensionality
    @assert length(output) == 2 "The NiceFlowLayer currently only supports 2 dimensional inputs and outputs."

    # fetch variables
    f = getf(layer)

    # determine result
    result = zeros(Float64, 2,2)
    result[1,1] = 1.0
    result[2,1] = -jacobian(f,output[1])
    result[2,2] = 1.0
    
    # return result
    return LowerTriangular(result)

end

det_jacobian(layer::NiceFlowLayer, input::Array{Float64,1}) = 1
absdet_jacobian(layer::NiceFlowLayer, input::Array{Float64,1}) = 1
logdet_jacobian(layer::NiceFlowLayer, input::Array{Float64,1}) = 0
logabsdet_jacobian(layer::NiceFlowLayer, input::Array{Float64,1}) = 0

detinv_jacobian(layer::NiceFlowLayer, output::Array{Float64,1}) = 1
absdetinv_jacobian(layer::NiceFlowLayer, output::Array{Float64,1}) = 1
logdetinv_jacobian(layer::NiceFlowLayer, output::Array{Float64,1}) = 0
logabsdetinv_jacobian(layer::NiceFlowLayer, output::Array{Float64,1}) = 0;


## NiceFlowModel methods
getlayers(model::NiceFlowModel)         = model.layers
getforward(model::NiceFlowModel)        = (x) -> forward(model, x)
getbackward(model::NiceFlowModel)       = (x) -> backward(model, x)
getjacobian(model::NiceFlowModel)       = (x) -> jacobian(model, x)
getinv_jacobian(model::NiceFlowModel)   = (x) -> inv_jacobian(model, x)

function forward(model::NiceFlowModel, input::Array{Float64,1})

    # fetch layers
    layers = getlayers(model)

    # allocate space for result
    input_new = copy(input)
    output = copy(input)
    
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

function backward(model::NiceFlowModel, output::Array{Float64,1})

    # fetch layers
    layers = getlayers(model)

    # allocate space for result
    output_new = copy(output)
    input = copy(output)
    
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

function jacobian(model::NiceFlowModel, input::Array{Float64,1})

    # fetch layers
    layers = getlayers(model)

    # allocate space for output
    input_new = copy(input)
    output = copy(input)
    J = zeros(Float64, 2,2)
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

function inv_jacobian(model::NiceFlowModel, output::Array{Float64,1})

    # fetch layers
    layers = getlayers(model)

    # allocate space for output
    output_new = copy(output)
    input = copy(output)
    J = zeros(Float64, 2,2)
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

det_jacobian(model::NiceFlowModel, input::Array{Float64,1}) = 1
absdet_jacobian(model::NiceFlowModel, input::Array{Float64,1}) = 1
logdet_jacobian(model::NiceFlowModel, input::Array{Float64,1}) = 0
logabsdet_jacobian(model::NiceFlowModel, input::Array{Float64,1}) = 0

detinv_jacobian(model::NiceFlowModel, output::Array{Float64,1}) = 1
absdetinv_jacobian(model::NiceFlowModel, output::Array{Float64,1}) = 1
logdetinv_jacobian(model::NiceFlowModel, output::Array{Float64,1}) = 0
logabsdetinv_jacobian(model::NiceFlowModel, output::Array{Float64,1}) = 0;

## TODO: create a forward-jacobian joint function that calculates both at the same time