export GammaAB

using Distributions

struct GammaAB{T}
    a :: T
    b :: T
end

GammaAB(a::Float64, b::Float64) = GammaAB{Float64}(a, b)

Distributions.mean(g::GammaAB) = g.a / g.b
Distributions.var(g::GammaAB)  = g.a / (g.b)^2
