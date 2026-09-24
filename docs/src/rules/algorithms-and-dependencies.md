# [Algorithms and dependencies](@id rules-algorithms)

A rule belongs to an algorithm, and the algorithm, with the node's factorisation, decides what
each rule consumes. Almost every node runs under the default algorithm and declares nothing.

## [The default scheme](@id rules-algorithms-default)

Under [`DefaultAlgorithm`](@ref), whether a rule is belief propagation, variational message
passing or their structured form follows from the factorisation, not from the rule. The engine
gives a rule the messages of the interfaces in its own cluster and the marginals of the other
clusters: with every interface in one cluster, a rule gets messages only, and with a mean-field
factorisation, marginals only. A deterministic node's clusters are always its output and the
joint over its inputs.

```@docs
AbstractAlgorithm
DefaultAlgorithm
DefaultAlgorithmExtension
MessagePassingRulesBase.default_algorithm
```

## [A node's own algorithm](@id rules-algorithms-own)

A node whose rules ignore the factorisation declares an algorithm of its own and the inputs of
each rule. The mixtures do: `NormalMixture` runs under `NormalMixtureVMP`, always variational,
and `Mixture` under `MixtureBP`, always over messages.

```julia
@define_factor_node(
    node = NormalMixture,
    type = Stochastic,
    interfaces = [:out, :switch, :m..., :p...],
    algorithm = NormalMixtureVMP,
    dependencies = [
        :out => (q[:switch], q[:p...], q[:m...]),
        :switch => (q[:out], q[:p...], q[:m...]),
        (:m, k) => (q[:out], q[:switch], q[:p][k]),
        (:p, k) => (q[:out], q[:switch], q[:m][k]),
    ],
)
```

The inputs of a target are subscribed to in the order they are declared, which in variational
message passing is the update schedule: here each component's precision is updated before its
mean. `@define_dependencies` declares the dependencies of another algorithm for an existing
node, and a subtype of [`DefaultAlgorithmExtension`](@ref) overrides some rules or dependencies
of the default for one model while inheriting the rest.

```@docs
@define_dependencies
MessagePassingRulesBase.DependenciesSpec
MessagePassingRulesBase.dependencies_spec
```

## [Purity](@id rules-algorithms-purity)

A rule is pure unless declared otherwise: its result depends only on its inputs and its
context, randomness included, since it comes from `ctx.rng`, which the caller owns. An algorithm
that carries state of its own is impure.

```@docs
MessagePassingRulesBase.ispure
```

## [Finding rules](@id rules-algorithms-registry)

A rule is found through Julia's method table, so a rule defined in any loaded package is found,
whichever module defined it. Each package also keeps a registry of what it defined, which is for
introspection only: listing a package's rules, checking them, and suggesting candidates when no
rule fits. When no rule fits, the lookup returns a `RuleNotFound`, which the caller reports with
the closest candidates.

```@docs
MessagePassingRulesBase.RuleSpec
MessagePassingRulesBase.RuleNotFound
MessagePassingRulesBase.RuleNotFoundError
MessagePassingRulesBase.find_message_rule
MessagePassingRulesBase.registered_rules
MessagePassingRulesBase.missing_services
MessagePassingRulesBase.check_rules
MessagePassingRulesBase.RuleIssue
MessagePassingRulesBase.check_rule_ambiguities
```
