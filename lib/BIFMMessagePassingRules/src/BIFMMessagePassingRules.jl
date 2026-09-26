"""
    BIFMMessagePassingRules

Backward-information-filter forward-marginal (BIFM) smoothing of a linear state-space model:

- [`BIFM`](@ref), a whole time slice, `znext = A zprev + B in` and `out = C znext`, as one
  deterministic node, under [`BIFMSmoother`](@ref), which carries `A`, `B` and `C` and which the
  model must give;
- [`BIFMHelper`](@ref), at the start of the chain, which turns the backward pass into the forward
  one.

The backward pass sends information-form messages, the forward pass the marginals themselves; on
a chain the result is the Rauch–Tung–Striebel smoother's. The rules take multivariate normal
messages only, and **the free energy of a model with BIFM is not supported**: asking for it
throws a [`BIFMFreeEnergyError`](@ref BIFMMessagePassingRules.BIFMFreeEnergyError).

# Examples

```jldoctest; setup = :(using BIFMMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily)
julia> result = @call_message_update_rule(
           node = BIFM, target = :zprev, algorithm = BIFMSmoother([1.0;;], [1.0;;], [1.0;;]),
           m = (
               out = MvNormalMeanPrecision([1.0], [1.0;;]),
               in = MvNormalMeanPrecision([0.0], [1.0;;]),
               znext = MvNormalWeightedMeanPrecision([0.0], [0.0;;]),
           ),
       );

julia> mean(getresult(result)) ≈ [1.0]
true
```
"""
module BIFMMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: AbstractAlgorithm
using BayesBase: TerminalProdArgument, promote_samplefloattype
using FastCholesky: cholinv
using LinearAlgebra: I, mul!

export BIFM, BIFMHelper, BIFMSmoother

include("bifm.jl")
include("bifm_helper.jl")

end
