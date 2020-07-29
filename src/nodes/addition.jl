export AdditionNode

using Distributions

# SA = StaticArrays
function AdditionNode()
    return FactorNode(+, (:in1, :in2, :out), ((1, 2, 3), ))
end

### Out ###

function rule(::typeof(+), ::Val{:out}, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, marginals::Nothing, meta) where T
    return mean(messages[1]) + mean(messages[2])
end

function rule(::typeof(+), ::Val{:out}, ::Marginalisation, messages::Tuple{Message{Normal{T}}, Message{Normal{T}}}, marginals::Nothing, meta) where T
    in1d = messages[1]
    in2d = messages[2]
    return Normal{T}(mean(in1d) + mean(in2d), sqrt(var(in1d) + var(in2d)))
end

function rule(::typeof(+), ::Val{:out}, ::Marginalisation, messages::Tuple{Message{Normal{T}}, Message{T}}, marginals::Nothing, meta) where T
    in1d = messages[1]
    in2v = messages[2]
    return Normal{T}(mean(in1d) + mean(in2v), std(in1d))
end

function rule(::typeof(+), ::Val{:out}, ::Marginalisation, messages::Tuple{Message{T}, Message{Normal{T}}}, marginals::Nothing, meta) where T
    in1v = messages[1]
    in2d = messages[2]
    return Normal{T}(mean(in2d) + mean(in1v), std(in2d))
end

### In 1 ###

function rule(::typeof(+), ::Val{:in1}, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, marginals::Nothing, meta) where T
    return mean(messages[2]) - mean(messages[1])
end

function rule(::typeof(+), ::Val{:in1}, ::Marginalisation, messages::Tuple{Message{Normal{T}}, Message{Normal{T}}}, marginals::Nothing, meta) where T
    in2d = messages[1]
    outd = messages[2]
    return Normal{T}(mean(outd) - mean(in2d), sqrt(var(outd) + var(in2d)))
end

function rule(::typeof(+), ::Val{:in1}, ::Marginalisation, messages::Tuple{Message{Normal{T}}, Message{T}}, marginals::Nothing, meta) where T
    in2d = messages[1]
    outv = messages[2]
    return Normal{T}(mean(outv) - mean(in2d), std(in2d))
end

function rule(::typeof(+), ::Val{:in1}, ::Marginalisation, messages::Tuple{Message{T}, Message{Normal{T}}}, marginals::Nothing, meta) where T
    in2v = messages[1]
    outd = messages[2]
    return Normal{T}(mean(outd) - mean(in2v), std(outd))
end

### In 2 ###

function rule(::typeof(+), ::Val{:in2}, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, marginals::Nothing, meta) where T
    return mean(messages[2]) - mean(messages[1])
end

function rule(::typeof(+), ::Val{:in2}, ::Marginalisation, messages::Tuple{Message{Normal{T}}, Message{Normal{T}}}, marginals::Nothing, meta) where T
    in1d = messages[1]
    outd = messages[2]
    return Normal{T}(mean(outd) - mean(in1d), sqrt(var(outd) + var(in1d)))
end

function rule(::typeof(+), ::Val{:in2}, ::Marginalisation, messages::Tuple{Message{Normal{T}}, Message{T}}, marginals::Nothing, meta) where T
    in1d = messages[1]
    outv = messages[2]
    return Normal{T}(mean(outv) - mean(in1d), std(in1d))
end

function rule(::typeof(+), ::Val{:in2}, ::Marginalisation, messages::Tuple{Message{T}, Message{Normal{T}}}, marginals::Nothing, meta) where T
    in1v = messages[1]
    outd = messages[2]
    return Normal{T}(mean(outd) - in1v, std(outd))
end
