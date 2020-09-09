export make_node, rule

using Distributions

function make_node(::typeof(+)) 
    return FactorNode(+, Deterministic, (:out, :in1, :in2), ((1, 2, 3), ), nothing)
end

function make_node(::typeof(+), out, in1, in2)
    node = make_node(+)
    connect!(node, :out, out)
    connect!(node, :in1, in1)
    connect!(node, :in2, in2)
    return node
end

### Out ###

function rule(::typeof(+), ::Type{ Val{:out} }, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, ::Nothing, ::Nothing) where { T <: Real }
    return mean(messages[1]) + mean(messages[2])
end

function rule(::typeof(+), ::Type{ Val{:out} }, ::Marginalisation, messages::Tuple{Message{Normal{T}}, Message{Normal{T}}}, ::Nothing, ::Nothing) where { T <: Real }
    in1d = messages[1]
    in2d = messages[2]
    return Normal{T}(mean(in1d) + mean(in2d), sqrt(var(in1d) + var(in2d)))
end

function rule(::typeof(+), ::Type{ Val{:out} }, ::Marginalisation, messages::Tuple{Message{Normal{T}}, Message{T}}, ::Nothing, ::Nothing) where { T <: Real }
    in1d = messages[1]
    in2v = messages[2]
    return Normal{T}(mean(in1d) + mean(in2v), std(in1d))
end

function rule(::typeof(+), ::Type{ Val{:out} }, ::Marginalisation, messages::Tuple{Message{T}, Message{Normal{T}}}, ::Nothing, ::Nothing) where { T <: Real }
    in1v = messages[1]
    in2d = messages[2]
    return Normal{T}(mean(in2d) + mean(in1v), std(in2d))
end

### In 1 ###

function rule(::typeof(+), ::Type{ Val{:in1} }, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, ::Nothing, ::Nothing) where { T <: Real }
    return mean(messages[1]) - mean(messages[2])
end

function rule(::typeof(+), ::Type{ Val{:in1} }, ::Marginalisation, messages::Tuple{Message{Normal{T}}, Message{Normal{T}}}, ::Nothing, ::Nothing) where { T <: Real }
    outd = messages[1]
    in2d = messages[2]
    return Normal{T}(mean(outd) - mean(in2d), sqrt(var(outd) + var(in2d)))
end

function rule(::typeof(+), ::Type{ Val{:in1} }, ::Marginalisation, messages::Tuple{Message{T}, Message{Normal{T}}}, ::Nothing, ::Nothing) where { T <: Real }
    outv = messages[1]
    in2d = messages[2]
    return Normal{T}(mean(outv) - mean(in2d), std(in2d))
end

function rule(::typeof(+), ::Type{ Val{:in1} }, ::Marginalisation, messages::Tuple{Message{Normal{T}}, Message{T}}, ::Nothing, ::Nothing) where { T <: Real }
    outd = messages[1]
    in2v = messages[2]
    return Normal{T}(mean(outd) - mean(in2v), std(outd))
end

## In 1 NVM 

function rule(::typeof(+), ::Type{ Val{:in1} }, ::Marginalisation, messages::Tuple{Message{T}, Message{NormalMeanVariance{T}}}, ::Nothing, ::Nothing) where { T <: Real }
    outv = messages[1]
    in2d = messages[2]
    return NormalMeanVariance{T}(mean(outv) - mean(in2d), var(in2d))
end

### In 2 ###

function rule(::typeof(+), ::Type{ Val{:in2} }, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, ::Nothing, ::Nothing) where { T <: Real }
    return mean(messages[1]) - mean(messages[2])
end

function rule(::typeof(+), ::Type{ Val{:in2} }, ::Marginalisation, messages::Tuple{Message{Normal{T}}, Message{Normal{T}}}, ::Nothing, ::Nothing) where { T <: Real }
    outd = messages[1]
    in1d = messages[2]
    return Normal{T}(mean(outd) - mean(in1d), sqrt(var(outd) + var(in1d)))
end

function rule(::typeof(+), ::Type{ Val{:in2} }, ::Marginalisation, messages::Tuple{Message{T}, Message{Normal{T}}}, ::Nothing, ::Nothing) where { T <: Real }
    outv = messages[1]
    in1d = messages[2]
    return Normal{T}(mean(outv) - mean(in1d), std(in1d))
end

function rule(::typeof(+), ::Type{ Val{:in2} }, ::Marginalisation, messages::Tuple{Message{Normal{T}}, Message{T}}, ::Nothing, ::Nothing) where { T <: Real }
    outd = messages[1]
    in1v = messages[2]
    return Normal{T}(mean(outd) - in1v, std(outd))
end
