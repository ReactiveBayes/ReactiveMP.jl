```@meta
DocTestSetup = :(using AutoregressiveMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily)
```

# AutoregressiveMessagePassingRules

The message passing rules of the autoregressive nodes, for Bayesian autoregressive processes
whose coefficients and noise precision are learnt with the states. Use [`AR`](@ref) when the
coefficients and the precision have priors of their own, a normal and a gamma, and
[`ConjugateAR`](@ref) when they share a normal-gamma prior, which makes their update conjugate.
The rules are [variational](@extref MessagePassingRulesBase glossary-vmp): a model using either
node gives the [algorithm](@extref MessagePassingRulesBase glossary-algorithm) [`ARVMP`](@ref)
and a [factorisation](@extref MessagePassingRulesBase glossary-factorisation).

```@docs
AutoregressiveMessagePassingRules
```

!!! info "Where these rules run"
    [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) runs these rules on a factor
    graph, which [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds from a
    [GraphPPL](https://github.com/ReactiveBayes/GraphPPL.jl) model. The examples here call the
    rules directly, as a test does.

## Example

A [message](@extref MessagePassingRulesBase glossary-message) towards the state `y` of a
univariate AR(1), from the message on the previous state and the
[marginals](@extref MessagePassingRulesBase glossary-marginal) of the coefficient and the
precision, run with
[`@call_message_update_rule`](@extref MessagePassingRulesBase.@call_message_update_rule):

```jldoctest
julia> result = @call_message_update_rule(
           node = AR, target = :y, algorithm = ARVMP(Univariate, 1, ARsafe()),
           m = (x = NormalMeanVariance(1.0, 1.0),),
           q = (θ = NormalMeanVariance(1.0, 1.0), γ = GammaShapeRate(1.0, 1.0)),
       );

julia> mean_var(getresult(result)) .≈ (0.5, 1.5)
(true, true)
```

## The site

- [AR](@ref ar-node): the node with the coefficients and the precision on edges of their own,
  its algorithm [`ARVMP`](@ref) and the two modes [`ARsafe`](@ref) and [`ARunsafe`](@ref);
- [ConjugateAR](@ref conjugate-ar-node): the node with them joint on one normal-gamma edge.
