import SpecialFunctions: loggamma

@node GammaInverse Stochastic [
    out, (α, aliases = [shape]), (θ, aliases = [scale])
]

# average energy := U[q] = -𝔼_X log pdf(X)
# NOTE: support is x ∈ (0,∞), so 𝔼x = mean(q_out) = mean(x) in θ / 𝔼x will not be undefined
# NOTE: negative sign in average energy has already been incorporated in the result
@average_energy GammaInverse (
    q_out::GammaInverse, q_α::PointMass, q_θ::PointMass
) = begin
    # α̂ and β̂ after δ-dirac: mean of pointmasses
    (α, θ) = (mean(q_α), mean(q_θ))
    return -α * log(θ) +
           loggamma(α) +
           (α + 1) * mean(log, q_out) +
           θ / mean(q_out)
end
