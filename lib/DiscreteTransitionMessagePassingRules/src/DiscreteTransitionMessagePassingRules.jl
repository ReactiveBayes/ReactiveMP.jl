"""
    DiscreteTransitionMessagePassingRules

[`DiscreteTransition`](@ref), `out ~ Categorical(A[:, in, T1, …, Tn])`: a transition between
categoricals through a tensor `A` of probabilities, conditioned on any number of categoricals
`T`, with `A` known (a `PointMass`) or learned (a `DirichletCollection`). It is the transition
and emission node of hidden Markov models. As a tensor node, each of its rules is one tensor
contraction, written once for any factorisation: belief propagation, mean-field, structured,
and joints of some of the `T`s. It runs under the default algorithm, with an average energy.

# Examples

```jldoctest; setup = :(using DiscreteTransitionMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily)
julia> result = @call_message_update_rule(
           node = DiscreteTransition, target = :in,
           m = (out = Categorical([1.0, 0.0]),), q = (a = PointMass([0.9 0.2; 0.1 0.8]),),
       );

julia> probvec(getresult(result)) ≈ [0.9, 0.2] / 1.1
true
```
"""
module DiscreteTransitionMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: rule_inputs, Target
using BayesBase: clamplog, tiny, huge, convert_paramfloattype
using ExponentialFamily: softmax!
using LinearAlgebra: normalize!

export DiscreteTransition

include("tensors.jl")
include("node.jl")
include("rules.jl")

end
