const NODE_KEYWORDS = (:node, :type, :interfaces, :algorithm, :dependencies, :static_inputs, :matched_groups, :min_group_length, :factorisation, :initial_messages)

"""
    @define_factor_node(
        node = ..., type = Stochastic | Deterministic, interfaces = [...],
        algorithm = ..., dependencies = [...], initial_messages = [...], static_inputs = ...,
        matched_groups = [...], min_group_length = ..., factorisation = ...,
    )

Declare a factor node: its interfaces, its kind, and the algorithm its rules run under. Rules
are then defined for it with [`@define_message_update_rule`](@ref),
[`@define_marginal_update_rule`](@ref) and [`@define_average_energy`](@ref), and an engine builds
the node in a graph from this declaration ([`nodespec`](@ref)). The macro takes keyword
arguments only; `node`, `type` and `interfaces` are required, the rest optional. An unknown or
repeated keyword is an error at definition time, naming the valid ones.

For a stochastic node without groups, the macro also defines [`nodefunction`](@ref)`(node)`, the
log-density `(; out, μ, v) -> logpdf(node(μ, v), out)`, which the rule verification and the
rule fallbacks use; `node` must then be callable as a distribution of the other interfaces.

# Required keywords

- `node`: what the node is, a type, `NormalMeanVariance`, or a function, `+`. The same value
  names it in every rule and in a graph.

- `type`: `Stochastic`, for a node with a density `f(out | inputs)`, or `Deterministic`, for
  `out = f(inputs)`. A deterministic node's clusters are always its output and the joint over its
  inputs, whatever the graph's factorisation.

- `interfaces`: the node's interfaces, a vector, the output first by convention:
  - `:μ`, a single interface. Names may contain underscores;
  - `:inputs...`, a group of any number of members, `(:inputs, 1)`, `(:inputs, 2)`, …, which a
    graph gives as a whole. A group may be empty unless `min_group_length` says otherwise;
  - `(:μ, aliases = [:mean])`, an interface with other names a graph may use for it.

# Optional keywords

- `algorithm`: the algorithm the node's rules run under unless a call or a graph asks for
  another, a type, instantiated with no arguments, or a value. Default:
  [`DefaultAlgorithm`](@ref)`()`, as it should be for almost every node: under it, whether a rule
  is belief propagation, variational message passing or their structured form follows from the
  factorisation. A node declares its own only when its rules ignore the factorisation, as a
  mixture's do.

- `dependencies`: what each rule consumes under that algorithm, in place of the default scheme.

$(DOC_DEPENDENCY_ENTRIES)

  Default: none, and every target follows the default scheme. Declarations for the node's other
  algorithms go in [`@define_dependencies`](@ref).

- `initial_messages`: messages an engine seeds on the node's inbound interfaces before inference,
  where the graph sets none, a vector of `:name => message` pairs:
  `initial_messages = [:in => NormalMeanPrecision(0.0, 100.0)]`. For a rule that depends on its
  own edge, which would otherwise wait forever. One per single interface; a group has none.
  Default: `[]`. See [`initial_messages`](@ref).

- `static_inputs`: how the node treats inputs connected to constants and data. `:none`, the
  default, treats them like any other input; `:fold` folds them into the node function, reached as
  [`getnodefn`](@ref)`(ctx.node, target)`, and every update waits until they are available. An
  engine builds such a node with its function, `factornode(…; nodefn = f)`. See
  [`static_inputs`](@ref).

- `matched_groups`: groups that must have as many members as each other, a vector of tuples of
  group names: `matched_groups = [(:m, :p)]` for a mixture whose means and precisions come in
  pairs. Default: `[]`. See [`matched_groups`](@ref).

- `min_group_length`: the fewest members every group may have, a non-negative integer:
  `min_group_length = 2` for a mixture of at least two components. Needs a group. Default: `1`.
  See [`min_group_length`](@ref).

- `factorisation`: `:meanfield` accepts only graphs that give every interface a cluster of its
  own; `:any`, the default, accepts every factorisation. Not for a deterministic node. See
  [`required_factorisation`](@ref).

An engine checks `matched_groups`, `min_group_length` and `factorisation` when it creates the
node, so a malformed graph is an error there rather than a rule silently reading fewer
components.

# Examples

```julia
@define_factor_node(node = NormalMeanVariance, type = Stochastic, interfaces = [:out, (:μ, aliases = [:mean]), (:v, aliases = [:var])])

@define_factor_node(node = +, type = Deterministic, interfaces = [:out, :in1, :in2])

# A node whose rules ignore the factorisation, and so declares an algorithm of its own and
# what each rule consumes; the precisions come before the means, the update schedule:
@define_factor_node(
    node             = NormalMixture,
    type             = Stochastic,
    interfaces       = [:out, :switch, :m..., :p...],
    algorithm        = NormalMixtureVMP,
    matched_groups   = [(:m, :p)],
    min_group_length = 2,
    factorisation    = :meanfield,
    dependencies     = [
        :out => (q[:switch], q[:p...], q[:m...]),
        :switch => (q[:out], q[:p...], q[:m...]),
        (:m, k) => (q[:out], q[:switch], q[:p][k]),
        (:p, k) => (q[:out], q[:switch], q[:m][k]),
    ],
)
```
"""
macro define_factor_node(args...)
    return esc(define_factor_node_expr(__module__, __source__, args))
end

function define_factor_node_expr(mod, source, args)
    keywords = parse_keywords("define_factor_node", args, NODE_KEYWORDS, (:node, :type, :interfaces))
    static_inputs = get(keywords, :static_inputs, QuoteNode(:none))
    (quoted_symbol(static_inputs) in (:none, :fold)) ||
        error("@define_factor_node: `static_inputs` must be :none or :fold, got `$static_inputs`")

    node = keywords[:node]
    type = keywords[:type]
    type in (:Stochastic, :Deterministic) ||
        error("@define_factor_node: `type` must be Stochastic or Deterministic, got `$type`")
    parsed = parse_interfaces(keywords[:interfaces])
    algorithm = get(keywords, :algorithm, :($DefaultAlgorithm))
    groups = [i.name for i in parsed if i.group]
    matched = parse_matched_groups(get(keywords, :matched_groups, :([])), groups)
    min_length = parse_min_group_length(get(keywords, :min_group_length, 1), groups)
    factorisation = quoted_symbol(get(keywords, :factorisation, QuoteNode(:any)))
    seeded = parse_initial_messages(get(keywords, :initial_messages, :([])), parsed)
    factorisation in (:any, :meanfield) ||
        error("@define_factor_node: `factorisation` must be :any or :meanfield, got `$(keywords[:factorisation])`")
    (factorisation === :meanfield && type === :Deterministic) &&
        error("@define_factor_node: a deterministic node's clusters are its output and the joint over its inputs, so it cannot require `factorisation = :meanfield`")

    base = MessagePassingRulesBase
    dispatch = gensym(:dispatch)
    spec = gensym(:nodespec)
    interface_specs = [:($InterfaceSpec($(QuoteNode(i.name)), $(i.group), $(Tuple(i.aliases)))) for i in parsed]

    body = quote
        $base.@define_registry
        const $dispatch = $node_dispatch_type($node)
        const $spec = $NodeSpec(
            $node, $(type === :Stochastic ? Stochastic() : Deterministic()),
            ($(interface_specs...),), $instantiate_algorithm($algorithm), $static_inputs,
            $matched, $min_length, $(QuoteNode(factorisation)), $seeded,
            $(QuoteNode(Symbol(something(source.file, :none)))), $(source.line),
        )
        $base.nodespec(::$dispatch) = $spec
        # A literal, so a rule body's `rule_inputs` folds to a constant.
        $base.interface_groups(::$dispatch) = $(Tuple(groups))
        $register!($REGISTRY_NAME, $spec)
    end

    if haskey(keywords, :dependencies)
        decl = gensym(:dependencies)
        declaration = dependency_declaration_expr("define_factor_node", node, :(typeof($spec.algorithm)), keywords[:dependencies], nothing)
        push!(body.args, :(const $decl = $validate_dependencies($spec, $declaration)))
        push!(body.args, :($base.dependencies_spec(::$dispatch, ::typeof($spec.algorithm)) = $decl))
        push!(body.args, :($register!($REGISTRY_NAME, $decl)))
    end

    if type === :Stochastic && !any(i -> i.group, parsed)
        names = [i.name for i in parsed]
        out, params = first(names), names[2:end]
        push!(body.args, :($base.nodefunction(::$dispatch) = (; $(names...)) -> $logpdf($node($(params...)), $out)))
    end
    push!(body.args, nothing)
    return body
end

function parse_matched_groups(ex, groups)
    (ex isa Expr && ex.head === :vect) ||
        error("@define_factor_node: `matched_groups` must be a vector of tuples of group names like `[(:m, :p)]`, got `$ex`")
    matched = map(ex.args) do item
        names = item isa Expr && item.head === :tuple ? map(quoted_symbol, item.args) : [nothing]
        (length(names) >= 2 && all(!isnothing, names) && allunique(names)) ||
            error("@define_factor_node: each entry of `matched_groups` names two or more distinct groups, like `(:m, :p)`; got `$item`")
        for name in names
            name in groups || error("@define_factor_node: `matched_groups` names `$name`, which is not a group; the groups are $(Tuple(groups))")
        end
        Tuple(names)
    end
    return Tuple(matched)
end

function parse_initial_messages(ex, parsed)
    (ex isa Expr && ex.head === :vect) ||
        error("@define_factor_node: `initial_messages` must be a vector of pairs like `[:in => message]`, got `$ex`")
    names = Symbol[]
    pairs = map(ex.args) do item
        (item isa Expr && item.head === :call && item.args[1] === :(=>) && length(item.args) == 3) ||
            error("@define_factor_node: an entry of `initial_messages` is `:name => message`, got `$item`")
        name = quoted_symbol(item.args[2])
        interface = findfirst(i -> i.name === name, parsed)
        interface === nothing && error("@define_factor_node: `initial_messages` names `$(something(name, item.args[2]))`, which is not an interface")
        parsed[interface].group && error("@define_factor_node: `$name` is a group; an initial message is declared for a single interface")
        name in names && error("@define_factor_node: `initial_messages` names `$name` more than once")
        push!(names, name)
        :($(QuoteNode(name)) => $(item.args[3]))
    end
    return Expr(:tuple, pairs...)
end

function parse_min_group_length(ex, groups)
    (ex isa Integer && ex >= 0) || error("@define_factor_node: `min_group_length` must be a non-negative integer, got `$ex`")
    (ex == 1 || !isempty(groups)) || error("@define_factor_node: `min_group_length` needs a group, and the node declares none")
    return Int(ex)
end

function parse_interfaces(ex)
    (ex isa Expr && ex.head === :vect) ||
        error("@define_factor_node: `interfaces` must be a vector like `[:out, :μ]`, got `$ex`")
    parsed = []
    for item in ex.args
        push!(parsed, parse_interface(item))
    end
    names = Symbol[]
    for interface in parsed
        interface.name in names && error("@define_factor_node: duplicate interface `$(interface.name)`")
        push!(names, interface.name)
    end
    for interface in parsed, alias in interface.aliases
        alias in names && error("@define_factor_node: alias `$alias` of `$(interface.name)` is also an interface name")
    end
    return parsed
end

function parse_interface(item)
    name = quoted_symbol(item)
    name === nothing || return (name = name, group = false, aliases = Symbol[])
    if item isa Expr && item.head === :... && length(item.args) == 1
        name = quoted_symbol(item.args[1])
        name === nothing || return (name = name, group = true, aliases = Symbol[])
    end
    if item isa Expr && item.head === :tuple && !isempty(item.args)
        inner = parse_interface(item.args[1])
        aliases = Symbol[]
        for option in item.args[2:end]
            (
                option isa Expr && option.head === :(=) && option.args[1] === :aliases &&
                    option.args[2] isa Expr && option.args[2].head === :vect
            ) ||
                error("@define_factor_node: interface options are `aliases = [:a, ...]`, got `$option`")
            for alias in option.args[2].args
                symbol = quoted_symbol(alias)
                symbol === nothing && error("@define_factor_node: an alias must be a symbol like `:mean`, got `$alias`")
                push!(aliases, symbol)
            end
        end
        return (name = inner.name, group = inner.group, aliases = aliases)
    end
    return error("@define_factor_node: an interface must be a symbol like `:out`, a group like `:inputs...`, or `(:μ, aliases = [...])`; got `$item`")
end
