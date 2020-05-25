export GaussianMeanVariance

using Distributions

function GaussianMeanVariance(::Type{T} = Float64; factorisation = SA[ [ 1, 2, 3 ] ]) where T
    return Node(Normal{T}, SA[ :mean, :variance, :value ], factorisation)
end

# Messages ordered as Tuple{ :mean, :variance }
function rule(::Type{ <: Normal{T}}, ::Val{:value}, messages::Tuple{Message{T}, Message{T}}, beliefs::Nothing, meta::Nothing) where T
    mean     = getdata(messages[1])
    variance = sqrt(getdata(messages[2]))
    return Message(Normal(mean, variance))
end
