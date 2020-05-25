export GaussianMeanVariance

import Distributions: Gamma

function GammaNode(::Type{T} = Float64; factorisation = SA[ SA[ 1, 2, 3 ] ]) where T
    return Node(Gamma{T}, SA[ :shape, :scale, :value ], factorisation)
end

# BP rule
function rule(::Type{ <: Gamma{T} }, ::Val{:value}, messages::Tuple{Message{T}, Message{T}}, beliefs::Nothing, meta) where { T <: Real }
    shape = getdata(messages[1])
    scale = getdata(messages[2])
    return Message(Gamma{T}(shape, scale))
end
