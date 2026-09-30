# [The ecosystem](@id ecosystem)

The engine defines no node and no rule. The nodes, their rules and the numerics they need are in
packages of their own. Each rule package depends on
[`MessagePassingRulesBase`](@extref MessagePassingRulesBase MessagePassingRulesBase) and never
on the engine. The engine finds rules through the base package's method table, which is global
across every loaded package, so loading a package is enough for the engine to find its rules.
Every package has its own documentation site, linked from its name below.

## [The foundations](@id ecosystem-foundations)

| Package | What it is |
|---|---|
| [`MessagePassingRulesBase`](@extref MessagePassingRulesBase MessagePassingRulesBase) | How nodes, rules, average energies, algorithms and dependencies are declared and found: [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node), [`@define_message_update_rule`](@extref MessagePassingRulesBase.@define_message_update_rule), [`find_message_rule`](@extref MessagePassingRulesBase.find_message_rule), and calling a rule by hand |
| [`MessagePassingRulesTestUtils`](@extref MessagePassingRulesTestUtils MessagePassingRulesTestUtils) | Testing rules: tables of cases, [`@test_message_update_rule`](@extref MessagePassingRulesTestUtils.@test_message_update_rule), verification against a node's definition, comparisons with a reference, and the rule-coverage gate |
| [`MessagePassingRulesApproximations`](@extref MessagePassingRulesApproximations MessagePassingRulesApproximations) | Numerics for propagating moments through functions: the unscented transform, linearisation, Gauss–Hermite cubature and Rauch–Tung–Striebel smoothing; no node |

## [The nodes](@id ecosystem-nodes)

| Package | Nodes |
|---|---|
| [`StandardMessagePassingRules`](@extref StandardMessagePassingRules StandardMessagePassingRules) | The standard nodes: the distributions (normal, gamma, Wishart, Bernoulli, categorical, …), arithmetic (`+`, `-`, `*` and `dot`), logic ([`AND`](@extref StandardMessagePassingRules.AND), [`OR`](@extref StandardMessagePassingRules.OR), [`NOT`](@extref StandardMessagePassingRules.NOT), [`IMPLY`](@extref StandardMessagePassingRules.IMPLY)) and the mixtures ([`Mixture`](@extref StandardMessagePassingRules.Mixture), [`NormalMixture`](@extref StandardMessagePassingRules.NormalMixture), [`GammaMixture`](@extref StandardMessagePassingRules.GammaMixture)) |
| [`DeltaMessagePassingRules`](@extref DeltaMessagePassingRules DeltaMessagePassingRules) | [`DeltaFn`](@extref DeltaMessagePassingRules.DeltaFn), a deterministic function of its inputs, under [`DeltaApproximation`](@extref DeltaMessagePassingRules.DeltaApproximation): unscented, linearised, or projected with CVI |
| [`GaussianCouplingMessagePassingRules`](@extref GaussianCouplingMessagePassingRules GaussianCouplingMessagePassingRules) | [`GaussianCoupling`](@extref GaussianCouplingMessagePassingRules.GaussianCoupling), the edge potential of Gaussian belief propagation |
| [`ProbitMessagePassingRules`](@extref ProbitMessagePassingRules ProbitMessagePassingRules) | [`Probit`](@extref ProbitMessagePassingRules.Probit), a binary output through the normal CDF, by expectation propagation |
| [`GCVMessagePassingRules`](@extref GCVMessagePassingRules GCVMessagePassingRules) | [`GCV`](@extref GCVMessagePassingRules.GCV), a normal whose log-variance is linear in its inputs |
| [`SoftDotMessagePassingRules`](@extref SoftDotMessagePassingRules SoftDotMessagePassingRules) | [`SoftDot`](@extref SoftDotMessagePassingRules.SoftDot), a dot product with Gaussian noise |
| [`AutoregressiveMessagePassingRules`](@extref AutoregressiveMessagePassingRules AutoregressiveMessagePassingRules) | [`AR`](@extref AutoregressiveMessagePassingRules.AR) and [`ConjugateAR`](@extref AutoregressiveMessagePassingRules.ConjugateAR), autoregressive processes, under [`ARVMP`](@extref AutoregressiveMessagePassingRules.ARVMP), which a model must give |
| [`ContinuousTransitionMessagePassingRules`](@extref ContinuousTransitionMessagePassingRules ContinuousTransitionMessagePassingRules) | [`ContinuousTransition`](@extref ContinuousTransitionMessagePassingRules.ContinuousTransition), a transition through a matrix built from a vector, under [`CTVMP`](@extref ContinuousTransitionMessagePassingRules.CTVMP) |
| [`PolyaMessagePassingRules`](@extref PolyaMessagePassingRules PolyaMessagePassingRules) | [`BinomialPolya`](@extref PolyaMessagePassingRules.BinomialPolya) and [`MultinomialPolya`](@extref PolyaMessagePassingRules.MultinomialPolya), Pólya-Gamma augmented regressions. **GPL-3 licensed**, through PolyaGammaHybridSamplers |
| [`BIFMMessagePassingRules`](@extref BIFMMessagePassingRules BIFMMessagePassingRules) | [`BIFM`](@extref BIFMMessagePassingRules.BIFM) and [`BIFMHelper`](@extref BIFMMessagePassingRules.BIFMHelper), a linear state-space time slice for backward-information-filter forward-marginal smoothing; no free energy |
| [`FlowMessagePassingRules`](@extref FlowMessagePassingRules FlowMessagePassingRules) | [`Flow`](@extref FlowMessagePassingRules.Flow), an invertible transformation, with its flow models, layers and [`PermutationMatrix`](@extref FlowMessagePassingRules.PermutationMatrix) |
| [`DiscreteTransitionMessagePassingRules`](@extref DiscreteTransitionMessagePassingRules DiscreteTransitionMessagePassingRules) | [`DiscreteTransition`](@extref DiscreteTransitionMessagePassingRules.DiscreteTransition), a categorical transition through a tensor with any number of conditioning categoricals |

Every package but PolyaMessagePassingRules is MIT licensed.

## [Using a package](@id ecosystem-using)

A model loads the packages of the nodes it uses, next to the engine or next to RxInfer. A model
with normal distributions and a nonlinear function, for example, loads
[`StandardMessagePassingRules`](@extref StandardMessagePassingRules StandardMessagePassingRules)
for the normal node and
[`DeltaMessagePassingRules`](@extref DeltaMessagePassingRules DeltaMessagePassingRules) for
[`DeltaFn`](@extref DeltaMessagePassingRules.DeltaFn). The engine then finds the rules of both.
[RxInfer's documentation](https://reactivebayes.github.io/RxInfer.jl/stable/) shows such models
in full.

A node whose algorithm carries settings, or has no default, receives its algorithm at activation
(see [The algorithm](@ref lib-activation-options-algorithm)). A new node and its rules can live in
a package of their own, on the same template. The package declares the node and its rules with
the base package's macros, as [the example node](@ref example-node) does, and tests them with
[`MessagePassingRulesTestUtils`](@extref MessagePassingRulesTestUtils MessagePassingRulesTestUtils).
[Your first node](@extref MessagePassingRulesBase tutorial-first-node) walks through it.
