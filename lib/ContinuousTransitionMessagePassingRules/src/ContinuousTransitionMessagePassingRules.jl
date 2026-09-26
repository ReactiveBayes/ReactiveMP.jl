"""
    ContinuousTransitionMessagePassingRules

The [`ContinuousTransition`](@ref) node, `y ~ N(f(a) x, W⁻¹)`: a linear Gaussian transition from
`x` to `y` through a matrix built from the vector `a`, with the noise precision `W`, where `a`
and `W` are learned alongside the states. Its rules are variational, mean-field or structured
`q(y, x)`, with average energies, and run under [`CTVMP`](@ref), which carries `f` and which the
model must give. A nonlinear `f` is linearised with ForwardDiff.

# Examples

```jldoctest; setup = :(using ContinuousTransitionMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase)
julia> result = @call_message_update_rule(
           node = ContinuousTransition, target = :y, algorithm = CTVMP(a -> reshape(a, 1, 1)),
           q = (x = MvNormalMeanCovariance([2.0], [1.0;;]), a = MvNormalMeanCovariance([3.0], [1.0;;]), W = Wishart(2, [0.5;;])),
       );

julia> m, W = mean_precision(getresult(result));

julia> m ≈ [6.0] && W ≈ [1.0;;]
true
```

See also [`ContinuousTransition`](@ref), [`CTransition`](@ref).
"""
module ContinuousTransitionMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: AbstractAlgorithm
using ExponentialFamily: WishartFast
using FastCholesky: cholinv
using LinearAlgebra: tr, logdet
import ForwardDiff
import StandardMessagePassingRules

export ContinuousTransition, CTransition, CTVMP

include("node.jl")
include("rules.jl")

end
