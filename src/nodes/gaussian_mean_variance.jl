export GaussianMeanVarianceNode

function GaussianMeanVarianceNode(::Type{T} = Float64; factorisation = ((1, 2, 3), )) where T
    return FactorNode(NormalMeanVariance{T}, (:mean, :variance, :value), factorisation)
end

function rule(::Type{ <: NormalMeanVariance }, ::Type{ Val{:value} }, ::Marginalisation, messages::Tuple{Message{ <: NormalMeanVariance{T} }, Message{T}}, ::Nothing, meta) where { T <: Real }
    return NormalMeanVariance(mean(messages[1]), var(messages[1]) + mean(messages[2]))
end