```@meta
CurrentModule = MessagePassingRulesBase
```

# [Glossary](@id glossary)

The terms the documentation of the ReactiveMP packages uses, in alphabetical order. Every site
links here the first time a page uses a term.

### [Algorithm](@id glossary-algorithm)

The value that selects which rules of a node run, and carries their parameters, such as the
number of cubature points of an approximation. Almost every node runs under
[`DefaultAlgorithm`](@ref), where the [factorisation](@ref glossary-factorisation) alone decides
whether a rule is belief propagation or variational message passing. An algorithm is not an
inference scheme. See [Algorithms and dependencies](@ref).

### [Average energy](@id glossary-average-energy)

A node's term of the [Bethe free energy](@ref glossary-bethe-free-energy): the expected negative
log-density of the node under the marginals of its clusters, ``U[q] = -\mathbb{E}_q[\log f]``.
Defined with [`@define_average_energy`](@ref).

### [Belief propagation](@id glossary-belief-propagation)

Message passing with exact messages, also called the sum-product algorithm. The message from a
node towards one of its variables integrates the node's function against the messages on its
other edges. It computes exact marginals on a graph without cycles. A rule is a belief
propagation rule when it takes only [messages](@ref glossary-message).

### [Bethe free energy](@id glossary-bethe-free-energy)

The objective that message passing minimises: the sum of the nodes'
[average energies](@ref glossary-average-energy) minus the entropies of the clusters and
variables. At its minimum, the marginals are the posterior (exactly on a tree, approximately
otherwise) and its value is an upper bound on ``-\log p(\text{data})``.

### [Cluster](@id glossary-cluster)

A set of a node's [interfaces](@ref glossary-interface) whose variables share one factor of the
approximate posterior. A node's clusters come from the [factorisation](@ref glossary-factorisation):
under `q(out, μ) q(v)`, a node with interfaces `out`, `μ` and `v` has the clusters `(out, μ)`
and `(v,)`. A rule takes [messages](@ref glossary-message) from its target's own cluster and
[marginals](@ref glossary-marginal) of the other clusters.

### [Default scheme](@id glossary-default-scheme)

What a rule receives under [`DefaultAlgorithm`](@ref): the messages on the other interfaces of
its target's cluster and the marginals of the other clusters. A node that declares nothing else
follows it. See [Algorithms and dependencies](@ref).

### [Dependencies](@id glossary-dependencies)

The inputs a node's rules take, target by target: `m[:x]` for a message, `q[:x]` for a marginal,
`q[:x, :y]` for a joint marginal. The [default scheme](@ref glossary-default-scheme) derives
them from the factorisation; a node's own algorithm declares them with
[`@define_dependencies`](@ref).

### [Deterministic node](@id glossary-deterministic-node)

A node whose output is a function of its inputs, `out = f(in...)`, such as `+`. Its clusters
are always its output and the joint over its inputs. See [`Deterministic`](@ref).

### [Expectation propagation](@id glossary-expectation-propagation)

A message passing scheme that approximates each message by projecting the corresponding
marginal onto a simpler family, usually by matching moments. The Probit node uses it.

### [Factor graph](@id glossary-factor-graph)

A graph of a probabilistic model's factorisation: [factor nodes](@ref glossary-factor-node) for
the factors, variables for the random quantities, and an edge wherever a factor depends on a
variable. Inference runs on it by passing messages along the edges.

### [Factor node](@id glossary-factor-node)

A factor of the model, such as a normal density or a sum. It is declared once with
[`@define_factor_node`](@ref), which names its [interfaces](@ref glossary-interface), and it
computes the messages towards its variables with its [rules](@ref glossary-rule).

### [Factorisation](@id glossary-factorisation)

How the approximate posterior splits into independent factors, for example
`q(x, y, z) = q(x, y) q(z)`. It gives each node its [clusters](@ref glossary-cluster). Without a
factorisation, every node has one cluster and inference is belief propagation; with every
variable in a cluster of its own, it is mean-field variational message passing.

### [Group](@id glossary-group)

An interface with any number of members, declared with a trailing `...`, as `in...`. A sum's
summands and a mixture's components are groups. A rule targets a member as `(:in, k)` and reads
a group as a tuple in member order.

### [In-place rule](@id glossary-in-place-rule)

A rule that writes its result into a buffer the caller gives it, to save allocations. Declared
with `inplace = true`.

### [Initial message](@id glossary-initial-message)

A message a node places on one of its edges before any rule has run, so that a rule reading that
edge can start. Declared with `initial_messages`.

### [Interface](@id glossary-interface)

A named edge of a factor node, such as `out`, `μ` or `v`. The first interface is the output by
convention. An interface may have aliases, other names a model can use for it.

### [Log scale](@id glossary-log-scale)

The logarithm of a message's normalising constant: a message is
``\exp(\text{log scale}) \cdot p(x)`` with ``p`` a normalised distribution. Summed over a graph,
log scales give the model's evidence. See [Log scales](@ref).

### [Marginal](@id glossary-marginal)

The current belief about a variable, or about the variables of a cluster jointly: the normalised
product of the messages arriving at them. Written `q(x)`, and `q[:x]` among a rule's inputs.

### [Mean field](@id glossary-mean-field)

The [factorisation](@ref glossary-factorisation) that puts every variable in a cluster of its
own, `q(x, y, z) = q(x) q(y) q(z)`. Every rule then takes only marginals.

### [Message](@id glossary-message)

What a node tells one of its variables about it, summarising the rest of the graph on the node's
side. Written `μ(x)`, and `m[:x]` among a rule's inputs.

### [Point mass](@id glossary-point-mass)

A distribution that puts all its probability on one value: how an observation or a constant
enters a rule. `PointMass(2.0)` comes from BayesBase.

### [Pushforward](@id glossary-pushforward)

The distribution of `f(x)` when `x` has a known distribution. A deterministic node's message
towards its output is the pushforward of the messages on its inputs.

### [Rule](@id glossary-rule)

A function that computes a message, a joint marginal or an average energy of a node from its
inputs. A rule is found by the node, its target, the [algorithm](@ref glossary-algorithm) and the
types of its inputs. See [Defining rules](@ref).

### [Scratch](@id glossary-scratch)

Working memory a rule keeps between calls, created on its first call. Declared with `scratch`.

### [Service](@id glossary-service)

A value a rule needs from whoever runs it, such as a random number generator (`rng`). A rule
declares the services it reads with `ctx = (:rng,)` and reads them from its
[`RuleContext`](@ref). See [The rule context](@ref).

### [Stochastic node](@id glossary-stochastic-node)

A node with a probability density over its interfaces, `f(out | in...)`, such as a normal
distribution. See [`Stochastic`](@ref).

### [Structured variational message passing](@id glossary-structured-vmp)

Variational message passing under a factorisation that keeps some variables together in one
cluster. Its rules take messages from their own cluster and marginals from the others.

### [Variational message passing](@id glossary-vmp)

Message passing that minimises the [Bethe free energy](@ref glossary-bethe-free-energy) under a
[factorisation](@ref glossary-factorisation). A message from a node towards `x` is
``\exp \mathbb{E}_q[\log f]``, the expectation taken under the marginals of the node's other
clusters. It handles models where exact messages have no closed form.
