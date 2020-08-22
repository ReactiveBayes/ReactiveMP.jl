export GammaABNode

function GammaABNode(::Type{T} = Float64; factorisation = ((1, 2, 3), )) where T
    return FactorNode(GammaAB{T}, (:a, :b, :value), factorisation)
end

function rule(::Type{ <: GammaAB{T} }, ::Type{ Val{:value} }, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, marginals::Nothing, meta) where { T <: Real }
    return GammaAB{T}(mean(messages[1]), mean(messages[2]))
end
