const NODE_KEYWORDS = (:node, :type, :interfaces, :algorithm, :dependencies)

"""
    @define_factor_node(node = ..., type = Stochastic | Deterministic, interfaces = [...], algorithm = ...)

Declare a factor node.

- `interfaces` lists names as symbols, `:out`; a variadic group as `:inputs...`; aliases
  as `(:μ, aliases = [:mean])`. Names may contain underscores.
- `algorithm` is the default rules run under, `BP()` when omitted. A type is instantiated
  with no arguments.

```julia
@define_factor_node(
    node       = Mixture,
    type       = Stochastic,
    interfaces = [:out, :switch, :inputs...],
    algorithm  = VMP,
)
```
"""
macro define_factor_node(args...)
    return esc(define_factor_node_expr(__module__, __source__, args))
end

function define_factor_node_expr(mod, source, args)
    keywords = parse_keywords("define_factor_node", args, NODE_KEYWORDS, (:node, :type, :interfaces))
    haskey(keywords, :dependencies) &&
        error("@define_factor_node: `dependencies` is not supported yet")

    node = keywords[:node]
    type = keywords[:type]
    type in (:Stochastic, :Deterministic) ||
        error("@define_factor_node: `type` must be Stochastic or Deterministic, got `$type`")
    parsed = parse_interfaces(keywords[:interfaces])
    algorithm = get(keywords, :algorithm, :($BP))

    base = MessagePassingRulesBase
    dispatch = gensym(:dispatch)
    spec = gensym(:nodespec)
    interface_specs = [:($InterfaceSpec($(QuoteNode(i.name)), $(i.group), $(Tuple(i.aliases)))) for i in parsed]

    body = quote
        $base.@define_registry
        const $dispatch = $node_dispatch_type($node)
        const $spec = $NodeSpec(
            $node, $(type === :Stochastic ? Stochastic() : Deterministic()),
            ($(interface_specs...),), $instantiate_algorithm($algorithm), nothing,
            $(QuoteNode(Symbol(something(source.file, :none)))), $(source.line),
        )
        $base.nodespec(::$dispatch) = $spec
        $register!($REGISTRY_NAME, $spec)
    end

    if type === :Stochastic && !any(i -> i.group, parsed)
        names = [i.name for i in parsed]
        out, params = first(names), names[2:end]
        push!(body.args, :($base.nodefunction(::$dispatch) = (; $(names...)) -> $logpdf($node($(params...)), $out)))
    end
    push!(body.args, nothing)
    return body
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
