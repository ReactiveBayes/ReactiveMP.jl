export make_node, rule

function MvNormalMeanCovarianceNode(::Type{T} = Float64; factorisation = ((1, 2, 3), )) where T
    return FactorNode(MvNormalMeanCovariance{T}, Stochastic, (:mean, :variance, :value), factorisation, nothing)
end

function make_node(::Type{ <: MvNormalMeanCovariance{T} }; factorisation = ((1, 2, 3), )) where T
    return MvNormalMeanCovarianceNode(T, factorisation = factorisation)
end

function make_node(::Type{ <: MvNormalMeanCovariance{T} }, mean, variance, value; factorisation = ((1, 2, 3), )) where T
    node = make_node(MvNormalMeanCovariance{T}, factorisation = factorisation)
    connect!(node, :mean, mean)
    connect!(node, :variance, variance)
    connect!(node, :value, value)
    return node
end