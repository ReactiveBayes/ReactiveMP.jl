import SpecialFunctions: loggamma

@node GammaInverse Stochastic [out, (α, aliases = [shape]), (θ, aliases = [scale])]

# average energy := U[q] = -𝔼_X log pdf(X)
# TODO one-liner
@average_energy GammaInverse (q_out::GammaInverse, q_α::PointMass, q_θ::PointMass) = begin
    # α̂ and β̂ after δ-dirac: mean of pointmasses
    (α, θ) = (mean(q_α), mean(q_θ))

    energy = 0
    energy += α * log(θ)
    energy -= loggamma(α)
    # 𝔼 log X
    energy -= (α + 1) * mean(log, q_out)
    # 𝔼 X
    # TODO: mean(q_out) ?= 0
    energy -= θ / mean(q_out)
    # minus sign in front of 𝔼
    # NOTE: you can also switch all signs above and remove this line
    energy *= -1
    return energy
end
