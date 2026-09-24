const BODY_SLOTS = (:output, :scratch, :algo, :ctx, :args, :ann)
const PREALLOCATE_SLOTS = (:algo, :ctx, :args)
const SCRATCH_SLOTS = PREALLOCATE_SLOTS

const MESSAGE_KEYWORDS = (:node, :target, :algorithm, :args, :body, :inplace, :preallocate, :scratch, :pure, :ctx)
const AVERAGE_ENERGY_KEYWORDS = (:node, :algorithm, :args, :body, :pure, :ctx)

"""
    @define_message_update_rule(node = ..., target = ..., args = (...), body = (...) -> ..., ...)

Define the rule for the message a node sends towards one of its interfaces.

- `target`: `:out`, or `(:m, k)` for member `k` of the group `m`; writing `k` binds the
  index in `body`.
- `args`: the inputs the rule consumes, in the spelling of its dependencies: `m[:μ]::T` for
  a message, `q[:μ]::T` for a marginal, `q[(:y, :x)]::T` or `q[:y, :x]::T` for a structural
  cluster (members in interface order; a group in a cluster means all its members jointly,
  so `q[(:in,)]` is the joint over the group `in`, and `(:in, 1)` is one member, so
  `q[:out, (:in, 1)]` is the joint of `out` and `in`'s first member), and for a group `m[:in...]::T` (all members), `m[:in][k]::T` (the
  target's own member) or `m[:in][!k]::T` (all but it). A group arrives as a tuple in member
  order with `nothing` where the selection leaves a member out, so `args.m[:in][k]` means
  member `k` whatever was selected. An omitted type is `Any`.
- `body`: an ordinary lambda over some of the slots `(output, scratch, algo, ctx, args, ann)`,
  named in that order: `args` holds the inputs, `algo` the algorithm value, `ctx` the
  [`RuleContext`](@ref), `ann` the annotations (read `ann.m[:out]`, write with
  [`annotate!`](@ref)), `output` the buffer of an in-place rule and `scratch` its working memory.
- `algorithm`: the algorithm type the rule runs under. Almost every rule omits it and
  belongs to its node's default, usually [`DefaultAlgorithm`](@ref), which requires the node
  to be declared before the rule is loaded. Naming one is for a rule switcher's or a node's
  own algorithm.
- `inplace = true` with `preallocate = (args) -> buffer` and a body taking `output` first.
- `scratch = (args) -> memory`: working memory, built from the inputs (and optionally `algo`
  and `ctx`, as `preallocate`) and given to the body as `scratch`. An engine keeps one per
  outbound stream and reuses it, so it is **write-before-read**: it carries nothing between
  calls, and the engine may keep, drop or rebuild it whenever it likes. It never leaves the
  rule, is never shared with another rule, and combines with `inplace`.
- `pure = false` marks a rule that has side effects, whatever its algorithm declares.
- `ctx = (:product, ...)`: the context services the rule needs.

```julia
@define_message_update_rule(
    node    = NormalMeanVariance,
    target = :out,
    args    = (m[:μ]::PointMass, m[:v]::PointMass),
    body    = (args) -> NormalMeanVariance(mean(args.m[:μ]), mean(args.m[:v])),
)
```
"""
macro define_message_update_rule(args...)
    return esc(define_rule_expr(:message, __source__, args))
end

"""
    @define_marginal_update_rule(node = ..., target = (:y, :x), args = (...), body = ...)

Define the rule for the marginal of a structural cluster. Takes the same keywords as
[`@define_message_update_rule`](@ref); `target` lists the cluster members in interface
order.
"""
macro define_marginal_update_rule(args...)
    return esc(define_rule_expr(:marginal, __source__, args))
end

"""
    @define_average_energy(node = ..., args = (...), body = ...)

Define a node's average energy. Takes the keywords of
[`@define_message_update_rule`](@ref) except `target`, `inplace`, `preallocate` and `scratch`.
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
    allowed = kind === :average_energy ? AVERAGE_ENERGY_KEYWORDS : MESSAGE_KEYWORDS
    required = kind === :average_energy ? (:node, :args, :body) : (:node, :target, :args, :body)
    keywords = parse_keywords(name, macroargs, allowed, required)

    node = keywords[:node]
    target, index_name = kind === :average_energy ? (nothing, nothing) : parse_target(name, kind, keywords[:target])
    inputs = parse_rule_args(name, keywords[:args], index_name)
    inplace = get(keywords, :inplace, false)
    inplace isa Bool || error("@$name: `inplace` must be `true` or `false`")
    pure = get(keywords, :pure, nothing)
    pure === nothing || pure isa Bool || error("@$name: `pure` must be `true` or `false`")
    services = parse_services(name, get(keywords, :ctx, :(())))

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
    index_arg = index_name === nothing ? () : (:($target_index($target_arg)),)

    prealloc_defs = []
    prealloc = nothing
    if inplace
        pre = keywords[:preallocate]
        pre_slots = parse_slots(name, pre, PREALLOCATE_SLOTS; what = "preallocate")
        user_pre = gensym(:preallocate)
        pre_args = [gensym(slot) for slot in PREALLOCATE_SLOTS]
        pre_passed = [pre_args[findfirst(==(slot), PREALLOCATE_SLOTS)] for slot in pre_slots]
        pre_target = gensym(:target)
        pre_index = index_name === nothing ? () : (:($target_index($pre_target)),)
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
        sc_index = index_name === nothing ? () : (:($target_index($sc_target)),)
        push!(scratch_defs, :(const $user_sc = $(append_parameter(sc, index_name))))
        scratch_fn = :(($(sc_args...), $sc_target) -> $user_sc($(sc_passed...), $(sc_index...)))
    end

    algorithm_type = haskey(keywords, :algorithm) ? :($algorithm_dispatch_type($(keywords[:algorithm]))) :
        :(typeof($default_algorithm($node)))
    signature = rule_signature(inputs)
    algorithm_sym, signature_sym, spec_sym = gensym(:algorithm), gensym(:signature), gensym(:rulespec)
    dispatch_sym = gensym(:dispatch)

    method = if kind === :message
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
            inplace = $inplace,
            pure = $pure,
            services = $services,
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
        return (:($Target{$(QuoteNode(symbol))}), nothing)
    end
    # A cluster: interfaces, and members of a group written `(:T, 1)`.
    is_member(a) = a isa Expr && a.head === :tuple && length(a.args) == 2 && quoted_symbol(a.args[1]) !== nothing && a.args[2] isa Integer
    if ex isa Expr && ex.head === :tuple && length(ex.args) >= 1 && all(a -> quoted_symbol(a) !== nothing || is_member(a), ex.args) &&
            (kind === :marginal || !(length(ex.args) == 2 && ex.args[2] isa Symbol))
        kind === :marginal ||
            error("@$name: `target` of a message rule is `:out` or `(:m, k)`, got `$ex`")
        members = Tuple(map(a -> quoted_symbol(a) !== nothing ? quoted_symbol(a) : (quoted_symbol(a.args[1]), Int(a.args[2])), ex.args))
        return (:($ClusterTarget{$members}), nothing)
    end
    if kind === :message && ex isa Expr && ex.head === :tuple && length(ex.args) == 2 &&
            quoted_symbol(ex.args[1]) !== nothing && ex.args[2] isa Symbol
        return (:($IndexedTarget{$(QuoteNode(quoted_symbol(ex.args[1])))}), ex.args[2])
    end
    shapes = kind === :marginal ? "a cluster like `(:y, :x)`" : "`:out` or `(:m, k)`"
    return error("@$name: `target` must be $shapes, got `$ex`")
end

function parse_services(name, ex)
    entries = ex isa Expr && ex.head === :tuple ? ex.args : [ex]
    services = Symbol[]
    for entry in entries
        symbol = quoted_symbol(entry)
        symbol === nothing && error("@$name: `ctx` lists services as symbols, like `ctx = (:product,)`; got `$entry`")
        symbol in CONTEXT_SERVICES ||
            error("@$name: unknown context service `$symbol`; valid services are $(join(("`$s`" for s in CONTEXT_SERVICES), ", "))")
        push!(services, symbol)
    end
    return Tuple(services)
end

# One entry of `args`: `m[:μ]::T`, `q[:y, :x]::T`, and for a group `m[:in...]::T` (all
# members), `m[:in][k]::T` (the target's own member) or `m[:in][!k]::T` (all but it).
function parse_rule_args(name, ex, index_name)
    entries = ex isa Expr && ex.head === :tuple ? ex.args : [ex]
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
    return inputs
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
