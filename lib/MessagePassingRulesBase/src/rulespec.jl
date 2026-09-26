"""
    InputSpec

One typed input a rule declares in its `args`, as a [`RuleSpec`](@ref) records it. Fields:

- `container::Symbol`: `:m` for a message, `:q` for a marginal;
- `key`: the interface's name, or a cluster's tuple of members;
- `selection::Symbol`: `:single` for an interface, `:cluster` for a joint, and for a group
  `:all` (`m[:in...]`), `:aligned` (`m[:in][k]`) or `:allbutself` (`m[:in][!k]`);
- `type`: the declared type, of each member for a group.
"""
struct InputSpec
    container::Symbol
    key::Union{Symbol, Tuple{Vararg{ClusterMember}}}
    selection::Symbol
    type::Any
end

"""
    RuleSpec

A rule as data, and the thing that runs: what a definition macro builds, what resolution
([`find_message_rule`](@ref) and its siblings) returns, and what [`execute_rule`](@ref) runs.
It has no type parameters, so resolution returns one concrete type wherever it can be
inferred. It shows itself at the REPL with its inputs, flags, source, file and line.

What it declares, read as fields:

- `kind`: `:message`, `:marginal` or `:average_energy`;
- `node`, `algorithm`: the node, and the algorithm type the rule is defined for;
- `target`: the target's type, a [`Target`](@ref), [`IndexedTarget`](@ref) or
  [`ClusterTarget`](@ref) type, or `Nothing` for an average energy;
- `inputs`: its typed inputs, [`InputSpec`](@ref)s; for a rule declared with `default` among its
  `args` (`default` is `true`), only the typed ones beside it;
- `inplace`, `pure`, `services`: whether it writes into a given buffer, whether it is pure
  (declared, or its algorithm's [`ispure`](@ref)), and the context services it declares;
- `logscale`: what a message rule declares about its result's log scale: `nothing` for none, a
  number, a function, or [`from_body`](@ref); `reads_logscale`: whether it reads its inbound
  messages' log scales;
- `source`, `file`, `line`: its body's source and where it was defined.

The other fields are internal: `signature`, the [`RuleArgs`](@ref) type it dispatches on;
`body`, `prealloc` and `scratch`, the macro-generated functions over the full slot lists, the
body over `(output, scratch, algo, ctx, args, ann, target)` and the others over
`(algo, ctx, args, target)`. Run a rule through [`execute_rule`](@ref) or a call, never by
calling them.
"""
struct RuleSpec
    kind::Symbol
    node::Any
    target::Any
    algorithm::Type
    signature::Type
    inputs::Tuple{Vararg{InputSpec}}
    body::Function
    prealloc::Union{Nothing, Function}
    scratch::Union{Nothing, Function}
    default::Bool
    inplace::Bool
    pure::Bool
    services::Tuple{Vararg{Symbol}}
    logscale::Any
    reads_logscale::Bool
    source::String
    file::Symbol
    line::Int
end

function RuleSpec(;
        kind::Symbol, node, target, algorithm::Type, signature::Type, body::Function,
        inputs::Tuple{Vararg{InputSpec}} = (),
        prealloc = nothing, scratch = nothing, default::Bool = false, inplace::Bool = false, pure::Union{Nothing, Bool} = nothing,
        services::Tuple{Vararg{Symbol}} = (), logscale = nothing, reads_logscale::Bool = false, source::AbstractString = "",
        file::Symbol = :none, line::Integer = 0,
    )
    kind in (:message, :marginal, :average_energy) ||
        throw(ArgumentError("rule kind must be :message, :marginal or :average_energy, got :$kind"))
    inplace && prealloc === nothing &&
        throw(ArgumentError("an in-place rule needs a `preallocate` function"))
    valid_logscale_declaration(logscale) ||
        throw(ArgumentError("a rule's `logscale` is a number, a function of its inputs or `from_body`, got $(repr(logscale))"))
    kind === :message || (logscale === nothing && !reads_logscale) ||
        throw(ArgumentError("only a message rule has a log scale; `logscale` and `reads_logscale` are for message rules"))
    effective = something(pure, algorithm <: AbstractAlgorithm ? ispure(algorithm) : true)
    return RuleSpec(
        kind, node, target, algorithm, signature, inputs, body, prealloc, scratch, default, inplace, effective,
        services, logscale, reads_logscale, String(source), file, Int(line),
    )
end

"""
    RuleNotFound

What resolution ([`find_message_rule`](@ref), [`find_marginal_rule`](@ref),
[`find_average_energy`](@ref)) returns when no rule matches; resolution never throws. Fields:
`kind` (`:message`, `:marginal` or `:average_energy`), `node`, `target` (`nothing` for an
average energy), `algorithm`, the value the call asked for, and `args`, the [`RuleArgs`](@ref)
that found nothing.

A caller that needs a rule reports it by throwing a [`RuleNotFoundError`](@ref), as every call
in this package does; an engine may first consult a rule fallback.
"""
struct RuleNotFound
    kind::Symbol
    node::Any
    target::Any
    algorithm::Any
    args::Any
end

"""
    find_message_rule(node, target, algorithm, args) -> Union{RuleSpec, RuleNotFound}

Resolve the message rule of `node` towards `target` under `algorithm` for the inputs `args`,
without running it.

Every [`@define_message_update_rule`](@ref) adds a method, so resolution is Julia's dispatch over
every loaded package; it never throws.

# Arguments
- `node`: the node, a type or a function, as declared;
- `target`: a [`Target`](@ref) or an [`IndexedTarget`](@ref);
- `algorithm`: the algorithm value, whose type selects the rules; for a
  [`DefaultAlgorithmExtension`](@ref) without a rule of its own, the default's rule is returned;
- `args`: the inputs, a [`RuleArgs`](@ref); their keys and types select the rule.

# Returns
The [`RuleSpec`](@ref), or a [`RuleNotFound`](@ref) that names the algorithm the call asked for.
Run the spec with [`rule_algorithm`](@ref)`(spec, algorithm)`, which differs from `algorithm`
for an inherited rule.

See also [`find_marginal_rule`](@ref), [`find_average_energy`](@ref), [`which_message_update_rule`](@ref).
"""
function find_message_rule(node, target, algorithm, args)
    algorithm isa DefaultAlgorithmExtension || return RuleNotFound(:message, node, target, algorithm, args)
    inherited = find_message_rule(node, target, DefaultAlgorithm(), args)
    return inherited isa RuleSpec ? inherited : RuleNotFound(:message, node, target, algorithm, args)
end

"""
    find_marginal_rule(node, cluster::ClusterTarget, algorithm, args) -> Union{RuleSpec, RuleNotFound}

Resolve the marginal rule of `node` for the structural `cluster` under `algorithm` for the
inputs `args`, a [`RuleArgs`](@ref), without running it. Every
[`@define_marginal_update_rule`](@ref) adds a method; it never throws. For a
[`DefaultAlgorithmExtension`](@ref) without a rule of its own, the default's rule is returned.

# Returns
The [`RuleSpec`](@ref), or a [`RuleNotFound`](@ref) that names the algorithm the call asked for.

See also [`find_message_rule`](@ref), [`which_marginal_update_rule`](@ref).
"""
function find_marginal_rule(node, cluster, algorithm, args)
    algorithm isa DefaultAlgorithmExtension || return RuleNotFound(:marginal, node, cluster, algorithm, args)
    inherited = find_marginal_rule(node, cluster, DefaultAlgorithm(), args)
    return inherited isa RuleSpec ? inherited : RuleNotFound(:marginal, node, cluster, algorithm, args)
end

"""
    find_average_energy(node, algorithm, args) -> Union{RuleSpec, RuleNotFound}

Resolve the average energy of `node` under `algorithm` for the marginals in `args`, a
[`RuleArgs`](@ref), without computing it. Every [`@define_average_energy`](@ref) adds a method; it
never throws. For a [`DefaultAlgorithmExtension`](@ref) without an energy of its own, the
default's is returned.

# Returns
The [`RuleSpec`](@ref), or a [`RuleNotFound`](@ref) whose `target` is `nothing`.

See also [`find_message_rule`](@ref), [`which_average_energy`](@ref).
"""
function find_average_energy(node, algorithm, args)
    algorithm isa DefaultAlgorithmExtension || return RuleNotFound(:average_energy, node, nothing, algorithm, args)
    inherited = find_average_energy(node, DefaultAlgorithm(), args)
    return inherited isa RuleSpec ? inherited : RuleNotFound(:average_energy, node, nothing, algorithm, args)
end

"""
    rule_algorithm(spec::RuleSpec, algorithm) -> AbstractAlgorithm

The algorithm value to run `spec` with, for a call made under `algorithm`: the call's own, or
[`DefaultAlgorithm`](@ref)`()` when `spec` was reached through a
[`DefaultAlgorithmExtension`](@ref)'s fallback to the default, so that a rule always receives the
algorithm it was written for. An engine calls it before [`execute_rule`](@ref), as the
`message_passing_*` functions do.
"""
@inline rule_algorithm(spec::RuleSpec, algorithm) = algorithm isa spec.algorithm ? algorithm : DefaultAlgorithm()

"""
    RuleNotFoundError(notfound::RuleNotFound)

The error every call in this package throws when no rule fits: the `call_*` functions and
macros, the `which_*` queries and the `message_passing_*` functions. It wraps the
[`RuleNotFound`](@ref) resolution returned.

Its message reads, from the top:

- the call: the kind of rule, the node, the target, the algorithm, and the inputs given, each as
  `m[:name]::Type` or `q[:name]::Type`;
- a diagnosis: *no rule exists for this node and target under any algorithm*; *a rule of this
  shape exists, but the input types do not fit* (a rule takes exactly these inputs, and some
  type differs); or *no rule consumes this set of inputs* (the rules take other inputs, or are
  under another algorithm);
- the near misses, every rule for the node and target, by file and line, each with a line per
  slot: `✓` or `✗` for the algorithm, then for each input the rule takes, what was given for it
  or `not provided`, and each input given that the rule does not take, `provided but not
  consumed`.

For a node `Shift` with interfaces `out` and `in` and a single rule, towards `out`:

```julia
julia> @call_message_update_rule(node = Shift, target = :in, m = (out = 1.0,))
ERROR: no message rule for Shift towards :in under DefaultAlgorithm() takes the inputs (m[:out]::Float64)
  no rule exists for this node and target under any algorithm
```
"""
struct RuleNotFoundError <: Exception
    notfound::RuleNotFound
end

"""
    execute_rule(spec::RuleSpec, output, algorithm, ctx, args, ann, target)
    execute_rule(spec::RuleSpec, output, scratch, algorithm, ctx, args, ann, target)

Run a resolved rule and return its bare result, building no [`RuleResult`](@ref): the engine's
entry point.

# Arguments
- `spec`: the rule, as resolution returned it;
- `output`: for an in-place rule, the buffer to write into, or `nothing` to have the rule
  preallocate one; ignored otherwise;
- `scratch`: the rule's working memory, which an engine builds once with [`rule_scratch`](@ref)
  and passes on every call; without it, or with `nothing`, a rule that declares scratch gets a
  fresh one;
- `algorithm`: the value to run under, [`rule_algorithm`](@ref)`(spec, algorithm)`;
- `ctx`: the [`RuleContext`](@ref); its services are not checked here, which is
  [`check_services`](@ref)'s job, once, when the rule is resolved;
- `args`: the [`RuleArgs`](@ref);
- `ann`: the annotations, a [`RuleAnnotations`](@ref) or a [`NoAnnotations`](@ref);
- `target`: the [`Target`](@ref), [`IndexedTarget`](@ref) or [`ClusterTarget`](@ref), or
  `nothing` for an average energy.

Nothing here catches exceptions: whatever a rule throws propagates to the caller. The rule's log
scale is not computed; an engine that tracks log scales calls
[`execute_rule_with_logscale`](@ref) instead.

A rule never sees a missing input. When any input is `missing`, an engine does not call the
rule at all, and does not run the annotation processors that follow a rule either; the
result is `missing`, carrying only the annotations written before the call.
"""
@inline execute_rule(spec::RuleSpec, output, algorithm, ctx, args, ann, target) =
    execute_rule(spec, output, nothing, algorithm, ctx, args, ann, target)

@inline execute_rule(spec::RuleSpec, output, scratch, algorithm, ctx, args, ann, target) =
    unwrap_result(spec, first(execute_rule_body(spec, output, scratch, algorithm, ctx, args, ann, target)))

"""
    execute_rule_with_logscale(spec::RuleSpec, output, scratch, algorithm, ctx, args, ann, target) -> Tuple

Run a resolved message rule and return `(result, logscale)`, for an engine that tracks log
scales. The arguments are [`execute_rule`](@ref)'s. The log scale is the one the rule declares, a
number or what its function or body computes, or an [`UndefinedLogScale`](@ref) naming the rule
for a rule that declares none. It does not check that `args` carries log scales for a rule that
reads them; [`check_reads_logscale`](@ref) does.
"""
@inline function execute_rule_with_logscale(spec::RuleSpec, output, scratch, algorithm, ctx, args, ann, target)
    raw, _ = execute_rule_body(spec, output, scratch, algorithm, ctx, args, ann, target)
    result = unwrap_result(spec, raw)
    return result, rule_logscale(spec, spec.logscale, raw, algorithm, ctx, args, target)
end

# The body's raw return, a `WithLogScale` for a rule declared `from_body`, and the scratch it ran with.
@inline function execute_rule_body(spec::RuleSpec, output, scratch, algorithm, ctx, args, ann, target)
    if spec.inplace && output === nothing
        output = spec.prealloc(algorithm, ctx, args, target)
    end
    if scratch === nothing
        scratch = rule_scratch(spec, algorithm, ctx, args, target)
    end
    return spec.body(output, scratch, algorithm, ctx, args, ann, target), scratch
end

"""
    rule_scratch(spec::RuleSpec, algorithm, ctx, args, target)

Build the working memory a rule declares with `scratch` from these inputs, or return `nothing`
for a rule that declares none. An engine builds it once per outbound stream and passes it to
every [`execute_rule`](@ref) of that rule. The rule writes it before reading it, so the engine may
rebuild it whenever it likes.
"""
@inline rule_scratch(spec::RuleSpec, algorithm, ctx, args, target) =
    spec.scratch === nothing ? nothing : spec.scratch(algorithm, ctx, args, target)

@inline function throw_if_not_found(spec)
    spec isa RuleNotFound && throw(RuleNotFoundError(spec))
    return spec
end

"""
    check_reads_logscale(spec::RuleSpec, args::RuleArgs) -> nothing

Check that `args` carries log scales when `spec` reads them (`reads_logscale = true`). Whoever
resolves a rule calls it before running the rule; the calls in this package do.

# Throws
`ArgumentError` naming the rule when it reads log scales and `args.logscale` is `nothing`: its
caller does not track them.
"""
@inline function check_reads_logscale(spec::RuleSpec, args)
    spec.reads_logscale && args.logscale === nothing && throw(
        ArgumentError(
            "the $(rule_heading(spec)) reads the log scales of its inbound messages, and none were given: " *
                "pass them to a call as `logscale = (name = value, ...)`, or have the engine track them (ReactiveMP's `logscales = true`)",
        ),
    )
    return nothing
end

# Run a resolved rule into a `RuleResult`.
@inline function run_rule(spec::RuleSpec, output, algorithm, ctx, args, ann, target)
    check_reads_logscale(spec, args)
    raw, scratch = execute_rule_body(spec, output, nothing, algorithm, ctx, args, ann, target)
    result = unwrap_result(spec, raw)
    logscale = spec.kind === :message ? rule_logscale(spec, spec.logscale, raw, algorithm, ctx, args, target) : nothing
    return RuleResult(result, logscale, spec, algorithm, ctx, scratch, args, outgoing_annotations(ann), target)
end

outgoing_annotations(ann::RuleAnnotations) = ann.out
outgoing_annotations(ann) = ann

# Parts of the `message_passing_*` docstrings that several share, written once and interpolated.

const DOC_MPR_COMMON_ARGS = rstrip(
    """
    - `algorithm`: the algorithm value to run under. For a [`DefaultAlgorithmExtension`](@ref)
      without a rule of its own, the default's rule runs, with `DefaultAlgorithm()`
      ([`rule_algorithm`](@ref)).
    - `args`: the inputs, a [`RuleArgs`](@ref); their keys and types select the rule.
    - `ctx`: the [`RuleContext`](@ref) the rule runs with. Default: an empty context. **Its services
      are not checked**: a service the rule declares and `ctx` does not supply reads as `nothing`
      inside the rule. Call [`check_services`](@ref)`(getrule(result), ctx)`, or check the spec
      from [`find_message_rule`](@ref) first, for an engine's guarantee.
    - `ann`: the rule's `ann`, a [`RuleAnnotations`](@ref); or, for a rule that only writes
      annotations, an [`AnnotationStore`](@ref) or a [`NoAnnotations`](@ref). Default:
      `NoAnnotations()`, dropping what the rule annotates.
    """
)

const DOC_MPR_THROWS = rstrip(
    """
    - [`RuleNotFoundError`](@ref) when no rule fits;
    - `ArgumentError` when the rule reads log scales and `args` carries none
      ([`check_reads_logscale`](@ref));
    - whatever the rule throws, unchanged.
    """
)

const DOC_MPR_INPLACE_THROWS = rstrip(
    """
    - `ArgumentError` when the rule has no in-place form;
    """
)

"""
    message_passing_rule(node, target, algorithm, args, ctx = RuleContext(), ann = NoAnnotations()) -> RuleResult

Resolve the message rule of `node` towards `target` and run it, allocating its result: the
positional, non-interactive form of [`call_message_update_rule`](@ref), for code that builds its
[`RuleArgs`](@ref) itself. An engine resolves once and calls [`execute_rule`](@ref) instead.

# Arguments
- `node`: the node, as declared with [`@define_factor_node`](@ref).
- `target`: a [`Target`](@ref) or an [`IndexedTarget`](@ref).
$(DOC_MPR_COMMON_ARGS)

# Returns
A [`RuleResult`](@ref): [`getresult`](@ref) is the message, [`getlogscale`](@ref) its log scale.

# Throws
$(DOC_MPR_THROWS)

# Examples

```jldoctest
julia> using MessagePassingRulesBase: Target, RuleArgs

julia> struct Shift end

julia> @define_factor_node(node = Shift, type = Deterministic, interfaces = [:out, :in])

julia> @define_message_update_rule(node = Shift, target = :out, args = (m[:in]::Real,), body = (args) -> args.m[:in] + 1)

julia> getresult(message_passing_rule(Shift, Target(:out), DefaultAlgorithm(), RuleArgs(m = (in = 1.0,))))
2.0
```
"""
@inline function message_passing_rule(node, target, algorithm, args, ctx = RuleContext(), ann = NoAnnotations())
    spec = throw_if_not_found(find_message_rule(node, target, algorithm, args))
    return run_rule(spec, nothing, rule_algorithm(spec, algorithm), ctx, args, ann, target)
end

"""
    message_passing_rule!(output, node, target, algorithm, args, ctx = RuleContext(), ann = NoAnnotations()) -> RuleResult

Resolve the in-place message rule of `node` towards `target` and run it into the buffer
`output`, which [`getresult`](@ref) then returns.

# Arguments
- `output`: the buffer the rule writes into, shaped as its `preallocate` builds it.
- `node`: the node, as declared with [`@define_factor_node`](@ref).
- `target`: a [`Target`](@ref) or an [`IndexedTarget`](@ref).
$(DOC_MPR_COMMON_ARGS)

# Returns
A [`RuleResult`](@ref): [`getresult`](@ref) is the message, [`getlogscale`](@ref) its log scale.

# Throws
$(DOC_MPR_INPLACE_THROWS)
$(DOC_MPR_THROWS)

See also [`message_passing_rule`](@ref).
"""
@inline function message_passing_rule!(output, node, target, algorithm, args, ctx = RuleContext(), ann = NoAnnotations())
    spec = throw_if_not_found(find_message_rule(node, target, algorithm, args))
    spec.inplace || throw(ArgumentError("the rule for $node towards $target has no in-place form"))
    return run_rule(spec, output, rule_algorithm(spec, algorithm), ctx, args, ann, target)
end

"""
    message_passing_marginalrule(node, cluster, algorithm, args, ctx = RuleContext(), ann = NoAnnotations()) -> RuleResult

Resolve the marginal rule of `node` for `cluster` and run it, allocating its result: the
positional form of [`call_marginal_update_rule`](@ref).

# Arguments
- `node`: the node, as declared with [`@define_factor_node`](@ref).
- `cluster`: a [`ClusterTarget`](@ref).
$(DOC_MPR_COMMON_ARGS)

# Returns
A [`RuleResult`](@ref): [`getresult`](@ref) is the joint marginal; its log scale is `nothing`.

# Throws
$(DOC_MPR_THROWS)

See also [`message_passing_marginalrule!`](@ref).
"""
@inline function message_passing_marginalrule(node, cluster, algorithm, args, ctx = RuleContext(), ann = NoAnnotations())
    spec = throw_if_not_found(find_marginal_rule(node, cluster, algorithm, args))
    return run_rule(spec, nothing, rule_algorithm(spec, algorithm), ctx, args, ann, cluster)
end

"""
    message_passing_marginalrule!(output, node, cluster, algorithm, args, ctx = RuleContext(), ann = NoAnnotations()) -> RuleResult

Resolve the in-place marginal rule of `node` for `cluster` and run it into the buffer `output`,
which [`getresult`](@ref) then returns.

# Arguments
- `output`: the buffer the rule writes into, shaped as its `preallocate` builds it.
- `node`: the node, as declared with [`@define_factor_node`](@ref).
- `cluster`: a [`ClusterTarget`](@ref).
$(DOC_MPR_COMMON_ARGS)

# Returns
A [`RuleResult`](@ref): [`getresult`](@ref) is the joint marginal; its log scale is `nothing`.

# Throws
$(DOC_MPR_INPLACE_THROWS)
$(DOC_MPR_THROWS)

See also [`message_passing_marginalrule`](@ref).
"""
@inline function message_passing_marginalrule!(output, node, cluster, algorithm, args, ctx = RuleContext(), ann = NoAnnotations())
    spec = throw_if_not_found(find_marginal_rule(node, cluster, algorithm, args))
    spec.inplace || throw(ArgumentError("the marginal rule for $node over $cluster has no in-place form"))
    return run_rule(spec, output, rule_algorithm(spec, algorithm), ctx, args, ann, cluster)
end

"""
    message_passing_average_energy(node, algorithm, args, ctx = RuleContext(), ann = NoAnnotations()) -> RuleResult

Resolve the average energy of `node` and compute it: the positional form of
[`call_average_energy`](@ref).

# Arguments
- `node`: the node, as declared with [`@define_factor_node`](@ref).
$(DOC_MPR_COMMON_ARGS)

# Returns
A [`RuleResult`](@ref): [`getresult`](@ref) is the energy, a number; its log scale and its target
are `nothing`.

# Throws
$(DOC_MPR_THROWS)
"""
@inline function message_passing_average_energy(node, algorithm, args, ctx = RuleContext(), ann = NoAnnotations())
    spec = throw_if_not_found(find_average_energy(node, algorithm, args))
    return run_rule(spec, nothing, rule_algorithm(spec, algorithm), ctx, args, ann, nothing)
end

"""
    missing_services(spec::RuleSpec, ctx::RuleContext) -> Tuple{Vararg{Symbol}}

The context services `spec` declares that `ctx` does not supply: the names `ctx` has no entry
for, in the order the rule declares them. An entry whose value is `nothing`, such as an unset
`matrix_correction`, is supplied. Meant to be checked once, when a node is set up, rather than on
every call.

```jldoctest
julia> using MessagePassingRulesBase: RuleContext, missing_services

julia> struct Noisy end

julia> @define_factor_node(node = Noisy, type = Stochastic, interfaces = [:out, :in])

julia> @define_message_update_rule(node = Noisy, target = :out, args = (m[:in]::Real,), ctx = (:rng, :scale), body = (ctx, args) -> args.m[:in])

julia> spec = which_message_update_rule(Noisy, :out; m = (in = 1.0,));

julia> missing_services(spec, RuleContext(rng = nothing))
(:scale,)
```
"""
missing_services(spec::RuleSpec, ctx::RuleContext) =
    filter(service -> !haskey(getfield(ctx, :services), service), spec.services)

"""
    check_services(spec::RuleSpec, ctx::RuleContext) -> nothing

Check that `ctx` supplies every context service `spec` declares. An engine calls it when it
resolves a rule, before running it, so a service nobody supplies is an error there rather than
a `nothing` inside the rule. It allocates nothing when every service is supplied.

# Throws
`ArgumentError` naming the rule and the services [`missing_services`](@ref) lists.

The interactive calls ([`call_message_update_rule`](@ref) and its siblings, the `@call_*` macros)
and the `message_passing_*` calls **do not check**: they run with whatever context the caller
passes, empty by default, and a service it lacks reads as `nothing`. Call `check_services`
first for the engine's guarantee.
"""
@inline function check_services(spec::RuleSpec, ctx::RuleContext)
    # Against the names, which the context's type fixes: `haskey` on the `NamedTuple` would box it.
    names = keys(getfield(ctx, :services))
    all(service -> service in names, spec.services) || throw_missing_services(spec, ctx)
    return nothing
end

# Kept out of line, so that building the message costs nothing on the path that passes.
@noinline function throw_missing_services(spec::RuleSpec, ctx::RuleContext)
    missing = missing_services(spec, ctx)
    throw(
        ArgumentError(
            "the $(rule_heading(spec)) needs the context $(length(missing) == 1 ? "service" : "services") " *
                "$(join(map(repr, missing), ", ")), which its context does not supply; " *
                "supply them in the context the rule is called with (ReactiveMP's activation option `context = (name = value, ...)`)",
        ),
    )
end


"""
    default_inputs_match(args::RuleArgs, ::Val{required}, ::Type{types}) -> Bool

Whether `args` holds each typed input a rule declared with `default` names beside it,
`(container, key, selection)`, with its type in `types`, for a single interface, a cluster or a
whole group. The guard the definition macro generates for such a rule, which returns a
[`RuleNotFound`](@ref) where it is `false`. Computed from the types alone, so it folds to a
constant.
"""
@generated function default_inputs_match(args::RuleArgs{M, Q}, ::Val{required}, ::Type{types}) where {M, Q, required, types}
    mnames, mtypes = M.parameters[1], M.parameters[2].parameters
    qnames, qtypes = Q.parameters[1], Q.parameters[2].parameters
    jkeys, jtypes = Q.parameters[3], Q.parameters[4].parameters
    function held(container, key, selection, type)
        if selection === :cluster
            position = findfirst(==(key), jkeys)
            return position !== nothing && jtypes[position] <: type
        end
        names, types = container === :m ? (mnames, mtypes) : (qnames, qtypes)
        position = findfirst(==(key), names)
        position === nothing && return false
        selection === :single && return types[position] <: type
        return types[position] <: Tuple && all(t -> t <: Union{Nothing, type}, types[position].parameters)
    end
    return all(((r, type),) -> held(r..., type), zip(required, types.parameters))
end

"""
    rule_inputs(node, container::Union{Messages, Marginals}) -> Tuple{Vararg{Pair}}

The inputs in a rule's `args.m` or `args.q` as a tuple of `key => value` pairs: an interface by
its name, a member of a group as `(group, k)`, and a joint by its key, as `(:out, (:T, 1))`.
Members a group does not deliver, `nothing` in its tuple, are left out. `node` tells which names
are groups. It is for a rule declared with `default`, whose body walks the inputs the
factorisation delivered; the pairs have known types and constant keys, so the walk is
type-stable.

```jldoctest
julia> using MessagePassingRulesBase: Messages, rule_inputs

julia> struct Tensor end

julia> @define_factor_node(node = Tensor, type = Stochastic, interfaces = [:out, :T...])

julia> rule_inputs(Tensor, Messages((out = 1.0, T = (2.0, nothing, 3.0))))
((:T, 1) => 2.0, (:T, 3) => 3.0, :out => 1.0)
```
"""
rule_inputs(node, container::Union{Messages, Marginals}) = rule_inputs(Val(interface_groups(node)), container)

# Generated from the container's type, so the pairs are a tuple of known types, with the keys
# as constants: a rule body walking them is type-stable.
@generated function rule_inputs(::Val{groups}, container::Union{Messages, Marginals}) where {groups}
    singles = container <: Messages ? :(container.values) : :(container.singles)
    names, types = container.parameters[1], container.parameters[2].parameters
    pairs = Any[]
    for (i, (key, type)) in enumerate(zip(names, types))
        if key in groups && type <: Tuple
            for (k, member) in enumerate(type.parameters)
                member === Nothing || push!(pairs, :($(QuoteNode((key, k))) => getfield(getfield($singles, $i), $k)))
            end
        else
            push!(pairs, :($(QuoteNode(key)) => getfield($singles, $i)))
        end
    end
    if container <: Marginals
        for (j, key) in enumerate(container.parameters[3])
            push!(pairs, :($(QuoteNode(key)) => getfield(container.joints, $j)))
        end
    end
    return Expr(:tuple, pairs...)
end
