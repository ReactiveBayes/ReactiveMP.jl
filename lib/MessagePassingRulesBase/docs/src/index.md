```@meta
CurrentModule = MessagePassingRulesBase
```

# MessagePassingRulesBase

MessagePassingRulesBase is the rule system of the ReactiveMP ecosystem. It declares factor
nodes and the rules they compute, finds the rule for a call, and runs it, all without an
inference engine. Use it to define a node and its rules, as every rule package does, to call a
rule by hand at the REPL or in a test, or, as an engine author, to look rules up and run them.

A rule is an ordinary Julia function of its inputs, dispatched on the node, the target and the
types of the incoming messages and marginals. Resolution is Julia's own dispatch, so a rule
defined in any loaded package is found. The package depends on BayesBase alone: no
distribution package, no engine.

```@docs
MessagePassingRulesBase
```

!!! info "Where these rules run"
    This package defines message passing rules; it does not build or run models. The
    [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) engine runs the rules on a
    factor graph, and [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds that graph from
    a model written with [GraphPPL](https://github.com/ReactiveBayes/GraphPPL.jl). The examples
    here call the rules directly, as a test or an interactive session does.

## A first node and rule

A deterministic node `out = in + c`, for a known shift `c`, with a rule for the message towards
`out`. The messages here are a small normal type of the example's own, a mean and a variance;
a real rule package uses ExponentialFamily's distributions.

```jldoctest overview
julia> using MessagePassingRulesBase

julia> struct Gauss   # a normal, by its mean and variance
           m::Float64
           v::Float64
       end

julia> struct Shift end

julia> @define_factor_node(node = Shift, type = Deterministic, interfaces = [:out, :in, :c])

julia> @define_message_update_rule(
           node = Shift, target = :out,
           args = (m[:in]::Gauss, m[:c]::Real),
           logscale = 0,
           body = (args) -> Gauss(args.m[:in].m + args.m[:c], args.m[:in].v),
       )

julia> result = @call_message_update_rule(node = Shift, target = :out, m = (in = Gauss(1.0, 2.0), c = 3.0));

julia> getresult(result)
Gauss(4.0, 2.0)

julia> getlogscale(result)
0
```

The node's declaration names its interfaces; the rule names its node, its target, the inputs
it takes with their types, and its body. The call runs the rule as an engine would and returns a
[`RuleResult`](@ref), the message together with its log scale and everything that produced it.
An engine such as ReactiveMP builds the node in a graph from the same declaration and runs the
same rule whenever its inputs change.

## The site

- [Defining nodes](@ref): [`@define_factor_node`](@ref), what a declaration records, and the
  queries that read it.
- [Defining rules](@ref): the three rule macros, targets, the inputs a rule receives, the slots
  of its body, in-place rules, scratch memory and annotations.
- [Algorithms and dependencies](@ref): how the factorisation selects belief propagation,
  variational message passing or their structured form under the default algorithm; a node's
  own algorithm, extensions of the default, parametrised algorithms,
  [`@define_dependencies`](@ref) and initial messages.
- [Log scales](@ref): what a rule declares about the normaliser of its message, and how a rule
  reads its inputs' log scales.
- [The rule context](@ref): the services a rule reads from `ctx`, and who checks them.
- [Calling rules](@ref): calling a rule by hand, reading its result, finding out why no rule
  fits, and resolving and running rules from code.
- [Inspecting rules](@ref): which rule a call would run, and listing, tabulating and checking
  the rules that exist.
- [Rule fallbacks](@ref): what an engine may send where no rule fits.
- [Internals](@ref): the calls an engine makes to run a resolved rule, and internal helpers.
