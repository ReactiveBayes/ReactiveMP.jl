export MvNormalGamma

# `MvNormalGamma` is defined as a distribution in ExponentialFamily; here we register it as a
# (prior) factor node so it can be used in models, e.g. `w ~ MvNormalGamma(μ, Λ, α, β)`.
@node MvNormalGamma Stochastic [out, μ, Λ, α, β]
