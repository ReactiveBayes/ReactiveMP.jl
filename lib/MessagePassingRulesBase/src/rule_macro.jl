const BODY_SLOTS = (:output, :scratch, :algo, :ctx, :args, :ann)
const PREALLOCATE_SLOTS = (:algo, :ctx, :args)
const SCRATCH_SLOTS = PREALLOCATE_SLOTS
const LOGSCALE_SLOTS = PREALLOCATE_SLOTS

const MESSAGE_KEYWORDS = (:node, :target, :algorithm, :args, :body, :inplace, :preallocate, :scratch, :pure, :ctx, :logscale, :reads_logscale)
const MARGINAL_KEYWORDS = (:node, :target, :algorithm, :args, :body, :inplace, :preallocate, :scratch, :pure, :ctx)
const AVERAGE_ENERGY_KEYWORDS = (:node, :algorithm, :args, :body, :pure, :ctx)

# Parts of the definition macros' docstrings that several share, written once and interpolated.

const DOC_RULE_NODE = rstrip(
    """
    - `node`: the node, as declared with [`@define_factor_node`](@ref): a type, `NormalMeanVariance`,
      or a function, `+`.
    """
)

const DOC_RULE_ARGS = rstrip(
    """
    - `args`: the inputs the rule consumes, a tuple of entries `container[key]::T`, the container
      `m` for a message or `q` for a marginal. A type left out is `Any`; the types are what the rule
      dispatches on. For a single interface:
      - `m[:μ]::T`, `q[:μ]::T`: the message or the marginal on `μ`;
      - `q[:y, :x]::T`, or `q[(:y, :x)]::T`: the joint marginal of a structural cluster, its members
        in interface order. A group in a cluster means all its members jointly: `q[(:in,)]` is the
        joint over the group `in`. `(:in, 1)`, with a literal index, is one member:
        `q[:out, (:in, 1)]` is the joint of `out` and `in`'s first member.

      For a group, whose value is a tuple in member order:
      - `m[:in...]::T`: every member, each of type `T`;
      - `m[:in][k]::T`: the target's own member, `k` being the name an indexed target binds;
      - `m[:in][!k]::T`: every member but the target's own.

      With a selection, the tuple keeps every position and holds `nothing` where the selection
      leaves a member out, so `args.m[:in][k]` is member `k` whatever was selected.

      `default` among the entries stands for whatever inputs the default scheme delivers under the
      graph's factorisation. The rule then takes them all, requires the typed entries beside
      `default`, and walks them with [`rule_inputs`](@ref): one rule for every factorisation.
    """
)

const DOC_RULE_BODY_SLOTS = rstrip(
    """
      Name only the slots the rule uses, in the order given; a slot that is misspelled, repeated
      or out of order is an error.
      - `output`: the buffer to write into, for an in-place rule only, and then first;
      - `scratch`: the rule's working memory, when it declares `scratch`;
      - `algo`: the algorithm value the rule runs under, and so its parameters;
      - `ctx`: the [`RuleContext`](@ref), whose services the rule reads as `ctx.name`;
      - `args`: the inputs, read as `args.m[:μ]`, `args.q[:y, :x]`, `args.m[:in][k]`;
      - `ann`: the annotations: those that arrived with the inputs, `ann.m[:μ]`, and the rule's own,
        written with [`annotate!`](@ref)`(ann, key, value)`.
    """
)

const DOC_RULE_ALGORITHM = rstrip(
    """
    - `algorithm`: the algorithm the rule runs under, a type, or a value whose type is used.
      Default: the node's own, [`default_algorithm`](@ref)`(node)`, usually
      [`DefaultAlgorithm`](@ref); the node must then be declared before the rule is loaded. Almost
      every rule leaves it out; naming one is for a rule switcher, a
      [`DefaultAlgorithmExtension`](@ref), or a node's own algorithm. With a parametric algorithm
      type `T`, `algorithm = T` matches every `T{…}`, while leaving it out binds the rule to the
      type of the node's default instance only.
    """
)

const DOC_RULE_CTX = rstrip(
    """
    - `ctx`: the context services the rule reads, a tuple of symbols, `ctx = (:rng,)` or
      `ctx = (:node, :matrix_correction)`. Any name is allowed, so a rule may need a service of its
      own. An engine checks that its context supplies each one when it resolves the rule
      ([`check_services`](@ref)); a call by hand does not. Default: `()`, none.
    """
)

const DOC_RULE_INPLACE = rstrip(
    """
    - `inplace`: `true` for a rule that writes its result into a buffer it is given rather than
      allocating one. It needs `preallocate`, and its `body` takes `output` first. Default: `false`.

    - `preallocate`: for an in-place rule, a function building the buffer from the inputs, over the
      slots `(algo, ctx, args)`: `preallocate = (args) -> similar(mean(args.m[:μ]))`. Allowed only
      with `inplace = true`.
    """
)

const DOC_RULE_SCRATCH = rstrip(
    """
    - `scratch`: working memory, a function building it from the inputs over the slots
      `(algo, ctx, args)`, given to the body as its `scratch` slot:
      `scratch = (args) -> (work = similar(mean(args.m[:μ])),)`. An engine keeps one per outbound
      stream and reuses it, so it is **write-before-read**: it carries nothing between calls, and
      the engine may keep, drop or rebuild it whenever it likes. It never leaves the rule, is never
      shared with another rule, keeps the rule pure, and combines with `inplace`. The body takes
      `scratch` exactly when this is given. Default: none.
    """
)

const DOC_RULE_PURE = rstrip(
    """
    - `pure`: `false` for a rule with side effects, `true` for a pure rule under an impure
      algorithm. Default: the algorithm's declaration, [`ispure`](@ref), `true` for almost every
      algorithm. A pure rule mutates neither its inputs nor state shared beyond one call, and draws
      randomness only from `ctx.rng`.
    """
)

"""
    @define_message_update_rule(
        node = ..., target = ..., args = (...), body = (...) -> ...,
        algorithm = ..., logscale = ..., reads_logscale = ..., ctx = (...),
        inplace = ..., preallocate = ..., scratch = ..., pure = ...,
    )

Define the rule for the message a node sends towards one of its interfaces. The macro takes
keyword arguments only; `node`, `target`, `args` and `body` are required, the rest optional.
An unknown or repeated keyword is an error at definition time, naming the valid ones.

The rule becomes a method of [`find_message_rule`](@ref): it is found for its node, target,
algorithm and the types of the inputs `args` names, from any module, and its module's registry
lists it for introspection ([`list_rules`](@ref), [`check_rules`](@ref)).

# Required keywords

$(DOC_RULE_NODE)

- `target`: the interface the message goes to:
  - `:out`, a single interface;
  - `(:m, k)`, any member of the group `m`. The name `k` is bound to the member's index, an
    `Int`, in `body` and in the `preallocate`, `scratch` and `logscale` functions, without being
    listed among their parameters; `args` can select by it.

$(DOC_RULE_ARGS)

- `body`: the rule itself, an ordinary lambda returning the message, whose parameters are some
  of the slots `(output, scratch, algo, ctx, args, ann)`.

$(DOC_RULE_BODY_SLOTS)

# Optional keywords

$(DOC_RULE_ALGORITHM)

- `logscale`: the message's log scale, the scalar with `message = exp(logscale) · result` for
  the normalised `result` the rule returns: a rule's result may stand for an unnormalised
  function, as a belief-propagation message does, and this is the log of its normaliser. One of:
  - a number: `logscale = 0`, or `logscale = loghalf` (StatsFuns'); an `Irrational` keeps the
    message's float type, while `-logtwo` would not, being a `Float64`;
  - a function of the inputs, over the slots `(algo, ctx, args)`, in that order:
    `logscale = (args) -> -log(abs(mean(args.m[:A])))`;
  - `from_body`: the body returns [`with_logscale`](@ref)`(result, logscale)`, for a log scale
    computed alongside the result.

  Default: none declared. The message's log scale is then an [`UndefinedLogScale`](@ref) naming
  the rule, which propagates through products; only [`require_logscale`](@ref) turns it into an
  error.

- `reads_logscale`: `true` if the rule reads the log scales of its inbound messages, as
  `args.logscale.m[:x]`. Its caller must then provide them: an engine does when it tracks log
  scales, and a call by hand takes them as `logscale = (...)`; without them the call is an error
  ([`check_reads_logscale`](@ref)). Default: `false`.

$(DOC_RULE_CTX)

$(DOC_RULE_INPLACE)

$(DOC_RULE_SCRATCH)

$(DOC_RULE_PURE)

# Examples

```julia
@define_message_update_rule(
    node     = NormalMeanVariance,
    target   = :out,
    args     = (m[:μ]::PointMass, m[:v]::PointMass),
    logscale = 0,
    body     = (args) -> NormalMeanVariance(mean(args.m[:μ]), mean(args.m[:v])),
)

# Towards any member of the group `m`, reading the aligned member of the group `p`;
# `k` is bound in the body without being listed:
@define_message_update_rule(
    node   = NormalMixture,
    target = (:m, k),
    args   = (q[:out]::Any, q[:switch]::Categorical, q[:p][k]::Any),
    body   = (args) -> … probvec(args.q[:switch])[k] …,
)
```
"""
macro define_message_update_rule(args...)
    return esc(define_rule_expr(:message, __source__, args))
end

"""
    @define_marginal_update_rule(
        node = ..., target = (:y, :x), args = (...), body = (...) -> ...,
        algorithm = ..., ctx = (...), inplace = ..., preallocate = ..., scratch = ..., pure = ...,
    )

Define the rule for the joint marginal of a structural cluster: the marginal of, say,
`(:y, :x)` a node computes from the messages on the cluster's members and the marginals of its
other interfaces. The macro takes keyword arguments only; `node`, `target`, `args` and `body`
are required, the rest optional. An unknown or repeated keyword is an error at definition time,
naming the valid ones.

The rule becomes a method of [`find_marginal_rule`](@ref), found for its node, cluster,
algorithm and input types, from any module. A marginal carries no log scale, so `logscale` and
`reads_logscale` are not accepted.

# Required keywords

$(DOC_RULE_NODE)

- `target`: the cluster, its members in interface order:
  - `(:y, :x)`, a cluster of interfaces;
  - `(:out, (:T, 1))`, with a member of a group written with a literal index;
  - a bare name, `target = members`: any cluster of the node, the name bound to the cluster's
    key in `body` and in the `preallocate` and `scratch` functions, without being listed among
    their parameters. Used with `default` in `args`, for one rule over every factorisation.

$(DOC_RULE_ARGS)

  A marginal rule typically reads the messages on the cluster's members, `m[:y]` and `m[:x]`,
  and the marginals of the node's other interfaces.

- `body`: the rule itself, an ordinary lambda returning the joint marginal, whose parameters
  are some of the slots `(output, scratch, algo, ctx, args, ann)`.

$(DOC_RULE_BODY_SLOTS)

# Optional keywords

$(DOC_RULE_ALGORITHM)

$(DOC_RULE_CTX)

$(DOC_RULE_INPLACE)

$(DOC_RULE_SCRATCH)

$(DOC_RULE_PURE)

# Example

```julia
@define_marginal_update_rule(
    node   = NormalMeanVariance,
    target = (:out, :μ),
    args   = (m[:out]::NormalMeanVariance, m[:μ]::NormalMeanVariance, q[:v]::PointMass),
    body   = (args) -> …,
)
```
"""
macro define_marginal_update_rule(args...)
    return esc(define_rule_expr(:marginal, __source__, args))
end

"""
    @define_average_energy(node = ..., args = (...), body = (...) -> ..., algorithm = ..., ctx = (...), pure = ...)

Define a node's average energy, `E_q[-log f]` under the marginals of its clusters: the node's
term of the Bethe free energy before the clusters' entropies are subtracted. The macro takes
keyword arguments only; `node`, `args` and `body` are required, the rest optional. An unknown
or repeated keyword is an error at definition time, naming the valid ones.

The energy becomes a method of [`find_average_energy`](@ref), found for its node, algorithm and
input types, from any module. It has no target, returns a number and has no log scale, so
`target`, `inplace`, `preallocate`, `scratch`, `logscale` and `reads_logscale` are not accepted.

# Required keywords

$(DOC_RULE_NODE)

- `args`: the marginals the energy reads, one per cluster of the node's factorisation, a tuple
  of entries `q[key]::T`. A type left out is `Any`; the types are what the energy dispatches on.
  - `q[:μ]::T`: the marginal of a single interface, a cluster of its own;
  - `q[:y, :x]::T`, or `q[(:y, :x)]::T`: the joint marginal of a structural cluster, its members
    in interface order; `q[(:in,)]` is the joint over the group `in`, and `(:in, 1)` one member;
  - `q[:in...]::T`: every member of a group, each a cluster of its own, as a tuple in member
    order.

  `default` among the entries stands for whatever clusters the graph's factorisation delivers;
  the energy then takes them all, requires the typed entries beside `default`, and walks them
  with [`rule_inputs`](@ref): one energy for every factorisation.

- `body`: the energy, an ordinary lambda returning a real number, whose parameters are some of
  the slots `(algo, ctx, args, ann)`, named in that order:
  - `algo`: the algorithm value, and so its parameters;
  - `ctx`: the [`RuleContext`](@ref), whose services the energy reads as `ctx.name`;
  - `args`: the marginals, read as `args.q[:μ]`, `args.q[:y, :x]`;
  - `ann`: the annotations that arrived with the marginals, `ann.q[:μ]`.

# Optional keywords

$(DOC_RULE_ALGORITHM)

$(DOC_RULE_CTX)

$(DOC_RULE_PURE)

# Example

```julia
@define_average_energy(
    node = NormalMeanVariance,
    args = (q[:out]::Any, q[:μ]::Any, q[:v]::Any),
    body = (args) -> begin
        m_out, v_out = mean_var(args.q[:out])
        m_μ, v_μ = mean_var(args.q[:μ])
        (log(2π) + mean(log, args.q[:v]) + mean(inv, args.q[:v]) * (v_out + v_μ + abs2(m_out - m_μ))) / 2
    end,
)
```
"""
macro define_average_energy(args...)
    return esc(define_rule_expr(:average_energy, __source__, args))
end

const RULE_MACRO_NAMES = Dict(
    :message => "define_message_update_rule",
    :marginal => "define_marginal_update_rule",
    :average_energy => "define_average_energy",
)

function define_rule_expr(kind, source, macroargs)
    name = RULE_MACRO_NAMES[kind]
    allowed = kind === :average_energy ? AVERAGE_ENERGY_KEYWORDS : kind === :marginal ? MARGINAL_KEYWORDS : MESSAGE_KEYWORDS
    required = kind === :average_energy ? (:node, :args, :body) : (:node, :target, :args, :body)
    keywords = parse_keywords(name, macroargs, allowed, required)

    node = keywords[:node]
    target, index_name, wildcard = kind === :average_energy ? (nothing, nothing, false) : parse_target(name, kind, keywords[:target])
    # A group target's index, or for a marginal rule over any cluster the cluster's key.
    index_fn = wildcard ? cluster_members : target_index
    inputs, has_default = parse_rule_args(name, keywords[:args], wildcard ? nothing : index_name)
    inplace = get(keywords, :inplace, false)
    inplace isa Bool || error("@$name: `inplace` must be `true` or `false`")
    pure = get(keywords, :pure, nothing)
    pure === nothing || pure isa Bool || error("@$name: `pure` must be `true` or `false`")
    services = parse_services(name, get(keywords, :ctx, :(())))
    reads_logscale = get(keywords, :reads_logscale, false)
    reads_logscale isa Bool || error("@$name: `reads_logscale` must be `true` or `false`")

    body = keywords[:body]
    slots = parse_slots(name, body, BODY_SLOTS)
    if inplace
        haskey(keywords, :preallocate) || error("@$name: `preallocate` is required when `inplace = true`")
        (!isempty(slots) && first(slots) === :output) ||
            error("@$name: an in-place rule's body must take `output` first")
    else
        :output in slots && error("@$name: the `output` slot requires `inplace = true`")
        haskey(keywords, :preallocate) && error("@$name: `preallocate` requires `inplace = true`")
    end
    has_scratch = haskey(keywords, :scratch)
    :scratch in slots && !has_scratch && error("@$name: the `scratch` slot requires a `scratch` keyword, `scratch = (args) -> memory`")
    has_scratch && !(:scratch in slots) && error("@$name: the rule declares `scratch` but its body does not take it; name `scratch` among its slots")

    base = MessagePassingRulesBase
    user_body = gensym(:body)
    adapter_args = [gensym(slot) for slot in BODY_SLOTS]
    target_arg = gensym(:target)
    passed = [adapter_args[findfirst(==(slot), BODY_SLOTS)] for slot in slots]
    index_arg = index_name === nothing ? () : (:($index_fn($target_arg)),)

    prealloc_defs = []
    prealloc = nothing
    if inplace
        pre = keywords[:preallocate]
        pre_slots = parse_slots(name, pre, PREALLOCATE_SLOTS; what = "preallocate")
        user_pre = gensym(:preallocate)
        pre_args = [gensym(slot) for slot in PREALLOCATE_SLOTS]
        pre_passed = [pre_args[findfirst(==(slot), PREALLOCATE_SLOTS)] for slot in pre_slots]
        pre_target = gensym(:target)
        pre_index = index_name === nothing ? () : (:($index_fn($pre_target)),)
        push!(prealloc_defs, :(const $user_pre = $(append_parameter(pre, index_name))))
        prealloc = :(($(pre_args...), $pre_target) -> $user_pre($(pre_passed...), $(pre_index...)))
    end

    scratch_defs = []
    scratch_fn = nothing
    if has_scratch
        sc = keywords[:scratch]
        sc_slots = parse_slots(name, sc, SCRATCH_SLOTS; what = "scratch")
        user_sc = gensym(:scratch)
        sc_args = [gensym(slot) for slot in SCRATCH_SLOTS]
        sc_passed = [sc_args[findfirst(==(slot), SCRATCH_SLOTS)] for slot in sc_slots]
        sc_target = gensym(:target)
        sc_index = index_name === nothing ? () : (:($index_fn($sc_target)),)
        push!(scratch_defs, :(const $user_sc = $(append_parameter(sc, index_name))))
        scratch_fn = :(($(sc_args...), $sc_target) -> $user_sc($(sc_passed...), $(sc_index...)))
    end

    logscale_defs = []
    logscale_fn = nothing
    if haskey(keywords, :logscale)
        declaration = keywords[:logscale]
        if declaration isa Expr && declaration.head === :->
            ls_slots = parse_slots(name, declaration, LOGSCALE_SLOTS; what = "logscale")
            user_ls = gensym(:logscale)
            ls_args = [gensym(slot) for slot in LOGSCALE_SLOTS]
            ls_passed = [ls_args[findfirst(==(slot), LOGSCALE_SLOTS)] for slot in ls_slots]
            ls_target = gensym(:target)
            ls_index = index_name === nothing ? () : (:($index_fn($ls_target)),)
            push!(logscale_defs, :(const $user_ls = $(append_parameter(declaration, index_name))))
            logscale_fn = :(($(ls_args...), $ls_target) -> $user_ls($(ls_passed...), $(ls_index...)))
        elseif declaration === :from_body
            logscale_fn = from_body
        else
            # A number, checked by `RuleSpec`.
            logscale_fn = declaration
        end
    end

    algorithm_type = haskey(keywords, :algorithm) ? :($algorithm_dispatch_type($(keywords[:algorithm]))) :
        :(typeof($default_algorithm($node)))
    # A rule over the default scheme's inputs takes any arguments for its node, target and
    # algorithm, and checks the typed ones beside `default` with a guard that folds at compile time.
    signature = has_default ? RuleArgs : rule_signature(inputs)
    algorithm_sym, signature_sym, spec_sym = gensym(:algorithm), gensym(:signature), gensym(:rulespec)
    dispatch_sym = gensym(:dispatch)
    required = :($Val(($((:(($(QuoteNode(i.container)), $(QuoteNode(i.key)), $(QuoteNode(i.selection)))) for i in inputs)...),)))
    required_types = :(Tuple{$((i.type for i in inputs)...)})

    method = if has_default
        target_sym, algo_arg, args_arg = gensym(:target), gensym(:algorithm), gensym(:args)
        notfound(k, t) = :($RuleNotFound($(QuoteNode(k)), $node, $t, $algo_arg, $args_arg))
        guarded(k, t) = :($default_inputs_match($args_arg, $required, $required_types) ? $spec_sym : $(notfound(k, t)))
        if kind === :message
            :($base.find_message_rule(::$dispatch_sym, $target_sym::$target, $algo_arg::$algorithm_sym, $args_arg::$RuleArgs) = $(guarded(:message, target_sym)))
        elseif kind === :marginal
            :($base.find_marginal_rule(::$dispatch_sym, $target_sym::$target, $algo_arg::$algorithm_sym, $args_arg::$RuleArgs) = $(guarded(:marginal, target_sym)))
        else
            :($base.find_average_energy(::$dispatch_sym, $algo_arg::$algorithm_sym, $args_arg::$RuleArgs) = $(guarded(:average_energy, nothing)))
        end
    elseif kind === :message
        :($base.find_message_rule(::$dispatch_sym, ::$target, ::$algorithm_sym, ::$signature_sym) = $spec_sym)
    elseif kind === :marginal
        :($base.find_marginal_rule(::$dispatch_sym, ::$target, ::$algorithm_sym, ::$signature_sym) = $spec_sym)
    else
        :($base.find_average_energy(::$dispatch_sym, ::$algorithm_sym, ::$signature_sym) = $spec_sym)
    end

    return quote
        $base.@define_registry
        const $dispatch_sym = $node_dispatch_type($node)
        const $algorithm_sym = $algorithm_type
        const $signature_sym = $signature
        const $user_body = $(append_parameter(body, index_name))
        $(prealloc_defs...)
        $(scratch_defs...)
        $(logscale_defs...)
        const $spec_sym = $RuleSpec(
            kind = $(QuoteNode(kind)),
            node = $node,
            target = $(target === nothing ? :Nothing : target),
            algorithm = $algorithm_sym,
            signature = $signature_sym,
            inputs = $(input_specs(inputs)),
            body = ($(adapter_args...), $target_arg) -> $user_body($(passed...), $(index_arg...)),
            prealloc = $prealloc,
            scratch = $scratch_fn,
            default = $has_default,
            inplace = $inplace,
            pure = $pure,
            services = $services,
            logscale = $logscale_fn,
            reads_logscale = $reads_logscale,
            source = $(string(MacroTools.prettify(body; alias = false))),
            file = $(QuoteNode(Symbol(something(source.file, :none)))),
            line = $(source.line),
        )
        $method
        $register!($REGISTRY_NAME, $spec_sym)
        nothing
    end
end

algorithm_dispatch_type(algorithm::Type) = algorithm
algorithm_dispatch_type(algorithm) = typeof(algorithm)

function parse_target(name, kind, ex)
    symbol = quoted_symbol(ex)
    if symbol !== nothing
        kind === :marginal &&
            error("@$name: `target` of a marginal rule is a cluster like `(:y, :x)`, got `$ex`")
        return (:($Target{$(QuoteNode(symbol))}), nothing, false)
    end
    # A bare name: a marginal rule over any cluster, the name bound to its key.
    if ex isa Symbol
        kind === :marginal ||
            error("@$name: `target` must be `:out` or `(:m, k)`, got `$ex`; a bare name is a marginal rule's, over any cluster")
        return (:($ClusterTarget), ex, true)
    end
    # A cluster: interfaces, and members of a group written `(:T, 1)`.
    is_member(a) = a isa Expr && a.head === :tuple && length(a.args) == 2 && quoted_symbol(a.args[1]) !== nothing && a.args[2] isa Integer
    if ex isa Expr && ex.head === :tuple && length(ex.args) >= 1 && all(a -> quoted_symbol(a) !== nothing || is_member(a), ex.args) &&
            (kind === :marginal || !(length(ex.args) == 2 && ex.args[2] isa Symbol))
        kind === :marginal ||
            error("@$name: `target` of a message rule is `:out` or `(:m, k)`, got `$ex`")
        members = Tuple(map(a -> quoted_symbol(a) !== nothing ? quoted_symbol(a) : (quoted_symbol(a.args[1]), Int(a.args[2])), ex.args))
        return (:($ClusterTarget{$members}), nothing, false)
    end
    if kind === :message && ex isa Expr && ex.head === :tuple && length(ex.args) == 2 &&
            quoted_symbol(ex.args[1]) !== nothing && ex.args[2] isa Symbol
        return (:($IndexedTarget{$(QuoteNode(quoted_symbol(ex.args[1])))}), ex.args[2], false)
    end
    shapes = kind === :marginal ? "a cluster like `(:y, :x)`" : "`:out` or `(:m, k)`"
    return error("@$name: `target` must be $shapes, got `$ex`")
end

function parse_services(name, ex)
    entries = ex isa Expr && ex.head === :tuple ? ex.args : [ex]
    services = Symbol[]
    for entry in entries
        symbol = quoted_symbol(entry)
        symbol === nothing && error("@$name: `ctx` lists services as symbols, like `ctx = (:rng,)`; got `$entry`")
        push!(services, symbol)
    end
    return Tuple(services)
end

# One entry of `args`: `m[:μ]::T`, `q[:y, :x]::T`, and for a group `m[:in...]::T` (all
# members), `m[:in][k]::T` (the target's own member) or `m[:in][!k]::T` (all but it).
function parse_rule_args(name, ex, index_name)
    entries = ex isa Expr && ex.head === :tuple ? ex.args : [ex]
    # `default` stands for the inputs the factorisation delivers, whatever they are.
    defaults = count(==(:default), entries)
    defaults > 1 && error("@$name: `default` twice in `args`")
    entries = filter(!=(:default), entries)
    inputs = []
    for entry in entries
        ref, type = entry isa Expr && entry.head === :(::) && length(entry.args) == 2 ?
            (entry.args[1], entry.args[2]) : (entry, :Any)
        (ref isa Expr && ref.head === :ref) ||
            error("@$name: each entry of `args` is `m[...]` or `q[...]`, like `m[:μ]::T`; got `$entry`")
        if ref.args[1] isa Expr && ref.args[1].head === :ref && length(ref.args) == 2
            push!(inputs, parse_member_selection(name, ref, type, index_name))
            continue
        end
        ref.args[1] in (:m, :q) ||
            error("@$name: each entry of `args` is `m[...]` or `q[...]`, like `m[:μ]::T`; got `$entry`")
        container, keys = ref.args[1], ref.args[2:end]
        isempty(keys) && error("@$name: `$ref` names no interface")
        if length(keys) == 1 && keys[1] isa Expr && keys[1].head === :tuple
            keys = keys[1].args
            isempty(keys) && error("@$name: `$ref` names no interface")
            push!(inputs, (container = container, key = cluster_members(name, ref, container, keys), selection = :cluster, type = type))
        elseif length(keys) == 1
            key = keys[1]
            if key isa Expr && key.head === :ref
                error("@$name: `$ref` indexes a Symbol; select member `k` of a group as `$(container)[:p][k]`")
            end
            group = key isa Expr && key.head === :... && length(key.args) == 1
            symbol = quoted_symbol(group ? key.args[1] : key)
            symbol === nothing && error("@$name: an interface in `args` is a symbol like `:μ`, got `$key`")
            push!(inputs, (container = container, key = symbol, selection = group ? :all : :single, type = type))
        else
            push!(inputs, (container = container, key = cluster_members(name, ref, container, keys), selection = :cluster, type = type))
        end
    end
    seen = Set()
    for input in inputs
        id = (input.container, input.key)
        id in seen && error("@$name: `$(input.container)[$(input.key)]` is given twice in `args`")
        push!(seen, id)
    end
    return inputs, defaults == 1
end

function cluster_members(name, ref, container, keys)
    container === :q || error("@$name: `$ref`: only marginals have structural clusters; use `q[...]`")
    members = map(keys) do key
        symbol = quoted_symbol(key)
        symbol === nothing || return symbol
        # One member of a group, by a literal index: `(:T, 1)`.
        if key isa Expr && key.head === :tuple && length(key.args) == 2 && quoted_symbol(key.args[1]) !== nothing
            key.args[2] isa Integer && return (quoted_symbol(key.args[1]), Int(key.args[2]))
            error("@$name: a group member in a cluster is `(:T, 1)`, with a literal index; got `$key`")
        end
        error("@$name: a cluster member is a symbol like `:y`, or a group member like `(:T, 1)`; got `$key`")
    end
    return Tuple(members)
end

function parse_member_selection(name, ref, type, index_name)
    inner, selector = ref.args[1], ref.args[2]
    container, key = inner.args[1], length(inner.args) == 2 ? quoted_symbol(inner.args[2]) : nothing
    (container in (:m, :q) && key !== nothing) ||
        error("@$name: a group member is selected as `m[:p][k]`, got `$ref`")
    index_name === nothing &&
        error("@$name: `$ref` selects by the target's index, which needs an indexed target like `target = (:$key, k)`")
    selection = if selector === index_name
        :aligned
    elseif selector isa Expr && selector.head === :call && selector.args == [:!, index_name]
        :allbutself
    else
        error("@$name: in `$ref`, a group member is selected by the target's index `$index_name`, as `[$index_name]` or `[!$index_name]`")
    end
    return (container = container, key = key, selection = selection, type = type)
end

# The `RuleArgs` type a rule dispatches on, with keys in the containers' canonical order. A
# group is a tuple in member order; members a selection leaves out are `nothing`.
function rule_signature(inputs)
    function element(input)
        input.selection === :all && return :(Tuple{Vararg{$(input.type)}})
        input.selection in (:aligned, :allbutself) && return :(Tuple{Vararg{Union{Nothing, $(input.type)}}})
        return input.type
    end
    function named(selected)
        sorted = sort(selected; by = input -> input.key)
        return Tuple(input.key for input in sorted), [element(input) for input in sorted]
    end
    mkeys, mtypes = named([i for i in inputs if i.container === :m])
    qkeys, qtypes = named([i for i in inputs if i.container === :q && i.selection !== :cluster])
    joints = [i for i in inputs if i.selection === :cluster]
    sorted = sort(joints; by = input -> cluster_sort_key(input.key))
    jkeys, jtypes = Tuple(input.key for input in sorted), [element(input) for input in sorted]
    return :(
        $RuleArgs{
            <:$Messages{$mkeys, <:Tuple{$(mtypes...)}},
            <:$Marginals{$qkeys, <:Tuple{$(qtypes...)}, $jkeys, <:Tuple{$(jtypes...)}},
        }
    )
end

input_specs(inputs) = Expr(
    :tuple,
    (:($InputSpec($(QuoteNode(i.container)), $(QuoteNode(i.key)), $(QuoteNode(i.selection)), $(i.type))) for i in inputs)...,
)

function lambda_parameters(name, ex; what = "body")
    (ex isa Expr && ex.head === :->) ||
        error("@$name: `$what` must be a lambda like `(args) -> ...`, got `$ex`")
    params = ex.args[1]
    params isa Symbol && return [params]
    (params isa Expr && params.head === :tuple) || return [params]
    return params.args
end

parameter_name(param) = param isa Symbol ? param :
    param isa Expr && param.head === :(::) ? param.args[1] : nothing

function parse_slots(name, ex, valid; what = "body")
    slots = Symbol[]
    for param in lambda_parameters(name, ex; what = what)
        slot = parameter_name(param)
        (slot isa Symbol && slot in valid) ||
            error("@$name: unknown $what slot `$param`; valid slots are $(join(("`$s`" for s in valid), ", ")), in that order")
        slot in slots && error("@$name: $what slot `$slot` is named twice")
        push!(slots, slot)
    end
    positions = [findfirst(==(slot), valid) for slot in slots]
    issorted(positions) ||
        error("@$name: $what slots must follow the canonical order $(join(valid, ", ")); got $(join(slots, ", "))")
    return slots
end

function append_parameter(lambda, parameter)
    parameter === nothing && return lambda
    params = lambda.args[1]
    list = params isa Symbol ? [params] : params isa Expr && params.head === :tuple ? copy(params.args) : [params]
    return Expr(:->, Expr(:tuple, list..., parameter), lambda.args[2])
end
