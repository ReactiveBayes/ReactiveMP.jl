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

An engine reads a declaration through its parts: each target's inputs, each input selecting an
interface, a whole group, a group's aligned member, every member but the target, or members a
function picks.

```@docs
MessagePassingRulesBase.TargetDependencies
MessagePassingRulesBase.target_dependencies
MessagePassingRulesBase.Dependency
MessagePassingRulesBase.DependencySelector
MessagePassingRulesBase.SingleInterface
MessagePassingRulesBase.AllGroupMembers
MessagePassingRulesBase.AlignedGroupMember
MessagePassingRulesBase.AllGroupMembersButSelf
MessagePassingRulesBase.CustomGroupSelector
MessagePassingRulesBase.select_group_members
MessagePassingRulesBase.selected_indices
MessagePassingRulesBase.selection_arity
MessagePassingRulesBase.free_energy_partition
```

## [Initial messages](@id rules-algorithms-initial-messages)

A rule that reads the message on its own edge, as an expectation-propagation rule does, has no
message to start from in a graph with a loop through that edge. The node may declare one, and the
engine sets it on the node's inbound message at activation, where nothing was set: a model's own
initialisation wins. Probit declares its v6 default for `in`:

```julia
@define_factor_node(
    node = Probit, type = Stochastic, interfaces = [:out, :in], algorithm = ProbitEP,
    initial_messages = [:in => NormalMeanPrecision(0.0, 100.0)],
)
```

It is a default for starting, not a dependency: which inputs a rule reads stays the algorithm's.

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
MessagePassingRulesBase.InputSpec
MessagePassingRulesBase.RuleNotFound
MessagePassingRulesBase.RuleNotFoundError
MessagePassingRulesBase.find_message_rule
MessagePassingRulesBase.find_marginal_rule
MessagePassingRulesBase.find_average_energy
MessagePassingRulesBase.rule_algorithm
MessagePassingRulesBase.missing_services
MessagePassingRulesBase.check_rules
MessagePassingRulesBase.RuleIssue
MessagePassingRulesBase.check_rule_ambiguities
```

The registries list what each package defined, for listings, coverage and display.

```@docs
MessagePassingRulesBase.Registry
MessagePassingRulesBase.registries
MessagePassingRulesBase.registered_rules
MessagePassingRulesBase.registered_nodes
MessagePassingRulesBase.registered_dependencies
MessagePassingRulesBase.list_rules
MessagePassingRulesBase.duplicate_rules
MessagePassingRulesBase.RuleCoverage
MessagePassingRulesBase.rule_coverage
MessagePassingRulesBase.visualize_spec
```
