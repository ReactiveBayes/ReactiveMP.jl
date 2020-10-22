export Gamma

import Distributions: Gamma, shape, scale

function prod(::ProdPreserveParametrisation, left::Gamma{T}, right::Gamma{T}) where T
    return Gamma(shape(left) + shape(right) - one(T), (scale(left) * scale(right)) / (scale(left) + scale(right)))
end