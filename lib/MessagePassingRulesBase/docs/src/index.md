```@meta
CurrentModule = MessagePassingRulesBase
```

# MessagePassingRulesBase

MessagePassingRulesBase is the rule system of the ReactiveMP ecosystem. It declares
[factor nodes](@ref glossary-factor-node) and the [rules](@ref glossary-rule) they compute. It
finds the rule for a call and runs it, without an inference engine.

This site is for rule authors, who define nodes and rules for a package of their own, and for
students who want to see how a message passing rule works by calling one. Engine authors find
the calls that look rules up and run them on the [Internals](@ref) page.

You use the package for three things:

- to define a node and its rules, as every rule package does;
- to call a rule by hand, at the REPL or in a test;
- to look rules up and run them, from an engine.

A rule is an ordinary Julia function of its inputs. Julia dispatches it on the node, on the
target and on the types of the incoming [messages](@ref glossary-message) and
[marginals](@ref glossary-marginal). Because resolution is Julia's own dispatch, a rule defined
in any loaded package is found.

The package depends on BayesBase and on small numerical packages (FastCholesky,
IrrationalConstants). It depends on no distribution package and on no engine. It also holds the
[math helpers](@ref math-helpers) that the rule packages share.

```@docs
MessagePassingRulesBase
```

!!! info "Where these rules run"
    The [ReactiveMP](https://reactivebayes.github.io/ReactiveMP.jl/dev/) engine runs these rules
    on a factor graph, which [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) builds from a
    model. The examples on this site call the rules directly, as a test does.

## A first node and rule

The node below is a [deterministic node](@ref glossary-deterministic-node), `out = in + c`, for
a known shift `c`. It has one rule, for the message towards `out`. The messages are a small
normal type that the example defines, a mean and a variance. A real rule package uses
ExponentialFamily's distributions instead.

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

The node's declaration names its [interfaces](@ref glossary-interface). The rule names its
node, its target, the inputs it takes with their types, and its body. The call runs the rule as
an engine would. It returns a [`RuleResult`](@ref): the message, its
[log scale](@ref glossary-log-scale) and everything that produced them. An engine such as
ReactiveMP builds the node in a graph from the same declaration. It runs the same rule whenever
the rule's inputs change.

[Your first node](@ref tutorial-first-node) builds a complete node step by step, with
ExponentialFamily's distributions.

## The site

- **Tutorials**: [Your first node](@ref tutorial-first-node) declares a normal node and writes
  its belief propagation and variational rules. [A deterministic node with a group](@ref tutorial-groups)
  writes rules for a sum of any number of inputs. [A node with its own algorithm](@ref tutorial-algorithm)
  gives a node a parametrised algorithm and declares what its rules take.
- [Defining nodes](@ref): [`@define_factor_node`](@ref), what a declaration records, and the
  queries that read it.
- [Defining rules](@ref): the three rule macros, targets, the inputs a rule receives, the slots
  of its body, in-place rules, scratch memory and annotations.
- [Algorithms and dependencies](@ref): how the factorisation selects belief propagation,
  variational message passing or their structured form under the default algorithm. The page
  also covers a node's own algorithm, extensions of the default, parametrised algorithms,
  [`@define_dependencies`](@ref) and initial messages.
- [Log scales](@ref): what a rule declares about the normalising constant of its message, and
  how a rule reads the log scales of its inputs.
- [The rule context](@ref): the services a rule reads from `ctx`, and who checks them.
- [Calling rules](@ref): how to call a rule by hand, read its result and find out why no rule
  fits, and how code resolves and runs rules.
- [Inspecting rules](@ref): which rule a call would run, and how to list, tabulate and check
  the rules that exist.
- [Rule fallbacks](@ref): what an engine may send where no rule fits.
- [Math helpers](@ref math-helpers): the linear algebra and Gaussian algebra the rule packages
  share.
- [Keyword reference](@ref keyword-reference): every keyword of every macro, in one place.
- [Glossary](@ref glossary): the terms these pages use.
- [Internals](@ref): the calls an engine makes to run a resolved rule, and internal helpers.
