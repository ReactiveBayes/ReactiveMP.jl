# [The ecosystem](@id ecosystem)

The engine defines no node and no rule. The nodes, their rules and the numerics they need are in
packages of their own, which depend on
[MessagePassingRulesBase](https://reactivebayes.github.io/MessagePassingRulesBase.jl/dev/) and
never on the engine: loading a package is enough for the engine to find its rules, since rules are
found through the base package's method table, global across every loaded package. Every package
has its own documentation site.

## [The foundations](@id ecosystem-foundations)

| Package | What it is |
|---|---|
| [MessagePassingRulesBase](https://reactivebayes.github.io/MessagePassingRulesBase.jl/dev/) | How nodes, rules, average energies, algorithms and dependencies are declared and found: [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node), [`@define_message_update_rule`](@extref MessagePassingRulesBase.@define_message_update_rule), [`find_message_rule`](@extref MessagePassingRulesBase.find_message_rule), and calling a rule by hand |
| [MessagePassingRulesTestUtils](https://reactivebayes.github.io/MessagePassingRulesTestUtils.jl/dev/) | Testing rules: tables of cases, [`@test_message_update_rule`](@extref MessagePassingRulesTestUtils.@test_message_update_rule), verification against a node's definition, comparisons with a reference, and the rule-coverage gate |
| [MessagePassingRulesApproximations](https://reactivebayes.github.io/MessagePassingRulesApproximations.jl/dev/) | Numerics for propagating moments through functions: the unscented transform, linearization, Gauss–Hermite cubature and Rauch–Tung–Striebel smoothing; no node |

## [The nodes](@id ecosystem-nodes)

| Package | Nodes |
|---|---|
| [StandardMessagePassingRules](https://reactivebayes.github.io/StandardMessagePassingRules.jl/dev/) | The standard nodes: the distributions (normal, gamma, Wishart, Bernoulli, categorical, …), arithmetic (`+`, `-`, `*` and `dot`), logic ([`AND`](@extref StandardMessagePassingRules.AND), [`OR`](@extref StandardMessagePassingRules.OR), [`NOT`](@extref StandardMessagePassingRules.NOT), [`IMPLY`](@extref StandardMessagePassingRules.IMPLY)) and the mixtures ([`Mixture`](@extref StandardMessagePassingRules.Mixture), [`NormalMixture`](@extref StandardMessagePassingRules.NormalMixture), [`GammaMixture`](@extref StandardMessagePassingRules.GammaMixture)) |
| [DeltaMessagePassingRules](https://reactivebayes.github.io/DeltaMessagePassingRules.jl/dev/) | [`DeltaFn`](@extref DeltaMessagePassingRules.DeltaFn), a deterministic function of its inputs, under [`DeltaApproximation`](@extref DeltaMessagePassingRules.DeltaApproximation): unscented, linearized, or projected with CVI |
| [GaussianCouplingMessagePassingRules](https://reactivebayes.github.io/GaussianCouplingMessagePassingRules.jl/dev/) | [`GaussianCoupling`](@extref GaussianCouplingMessagePassingRules.GaussianCoupling), the edge potential of Gaussian belief propagation |
| [ProbitMessagePassingRules](https://reactivebayes.github.io/ProbitMessagePassingRules.jl/dev/) | [`Probit`](@extref ProbitMessagePassingRules.Probit), a binary output through the normal CDF, by expectation propagation |
| [GCVMessagePassingRules](https://reactivebayes.github.io/GCVMessagePassingRules.jl/dev/) | [`GCV`](@extref GCVMessagePassingRules.GCV), a normal whose log-variance is linear in its inputs |
| [SoftDotMessagePassingRules](https://reactivebayes.github.io/SoftDotMessagePassingRules.jl/dev/) | [`SoftDot`](@extref SoftDotMessagePassingRules.SoftDot), a dot product with Gaussian noise |
| [AutoregressiveMessagePassingRules](https://reactivebayes.github.io/AutoregressiveMessagePassingRules.jl/dev/) | [`AR`](@extref AutoregressiveMessagePassingRules.AR) and [`ConjugateAR`](@extref AutoregressiveMessagePassingRules.ConjugateAR), autoregressive processes, under [`ARVMP`](@extref AutoregressiveMessagePassingRules.ARVMP), which a model must give |
| [ContinuousTransitionMessagePassingRules](https://reactivebayes.github.io/ContinuousTransitionMessagePassingRules.jl/dev/) | [`ContinuousTransition`](@extref ContinuousTransitionMessagePassingRules.ContinuousTransition), a transition through a matrix built from a vector, under [`CTVMP`](@extref ContinuousTransitionMessagePassingRules.CTVMP) |
| [PolyaMessagePassingRules](https://reactivebayes.github.io/PolyaMessagePassingRules.jl/dev/) | [`BinomialPolya`](@extref PolyaMessagePassingRules.BinomialPolya) and [`MultinomialPolya`](@extref PolyaMessagePassingRules.MultinomialPolya), Pólya-Gamma augmented regressions. **GPL-3 licensed**, through PolyaGammaHybridSamplers |
| [BIFMMessagePassingRules](https://reactivebayes.github.io/BIFMMessagePassingRules.jl/dev/) | [`BIFM`](@extref BIFMMessagePassingRules.BIFM) and [`BIFMHelper`](@extref BIFMMessagePassingRules.BIFMHelper), a linear state-space time slice for backward-information-filter forward-marginal smoothing; no free energy |
| [FlowMessagePassingRules](https://reactivebayes.github.io/FlowMessagePassingRules.jl/dev/) | [`Flow`](@extref FlowMessagePassingRules.Flow), an invertible transformation, with its flow models, layers and [`PermutationMatrix`](@extref FlowMessagePassingRules.PermutationMatrix) |
| [DiscreteTransitionMessagePassingRules](https://reactivebayes.github.io/DiscreteTransitionMessagePassingRules.jl/dev/) | [`DiscreteTransition`](@extref DiscreteTransitionMessagePassingRules.DiscreteTransition), a categorical transition through a tensor with any number of conditioning categoricals |

Every package but PolyaMessagePassingRules is MIT licensed.

## [Using a package](@id ecosystem-using)

A model loads the packages of the nodes it uses, next to the engine, or next to RxInfer:

```julia
using ReactiveMP, StandardMessagePassingRules, DeltaMessagePassingRules
```

A node whose algorithm carries settings, or has no default, is given its algorithm at activation
(see [The algorithm](@ref lib-activation-options-algorithm)). A new node and its rules can live in
a package of their own, on the same template: its declaration and rules with the base package's
macros, and its tests with the test tools.
