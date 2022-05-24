# Extra file for fast-fixes for packages ReactiveMP.jl depends on

# Hotfix for https://github.com/JuliaStats/Distributions.jl/issues/1270
function score(::DifferentialEntropy, marginal::Marginal{<:Dirichlet})
    d  = getdata(marginal)
    α  = d.alpha
    α0 = d.alpha0
    k  = length(α)
    en = d.lmnB + (α0 - k) * digamma(α0) - sum(αj -> (αj - 1) * digamma(αj), α)
end
