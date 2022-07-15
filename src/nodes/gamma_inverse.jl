import SpecialFunctions: loggamma

@node GammaInverse Stochastic [out, (α, aliases = [shape]), (β, aliases = [scale])]

# average energy := U[q] = -𝔼_X log pdf(X)
@average_energy GammaInverse (q_out::InverseGamma, q_α::PointMass, q_β::PointMass) = begin
    # α̂ and β̂ after δ-dirac: mean of pointmasses
    (α, β) = (mean(q_α), mean(q_β))

    energy = 0
    energy += α * log(β)
    energy -= loggamma(α)
    # 𝔼 log X
    energy -= (α + 1) * mean(log, q_out)
    # 𝔼 X
    # TODO: mean(q_out) ?= 0
    energy -= β / mean(q_out)
    # minus sign in front of 𝔼
    # NOTE: you can also switch all signs above and remove this line
    energy *= -1
    return energy
end
