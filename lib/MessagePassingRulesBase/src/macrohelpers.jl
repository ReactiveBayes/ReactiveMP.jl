function parse_keywords(macroname, args, allowed, required)
    keywords = Dict{Symbol, Any}()
    for arg in args
        arg isa LineNumberNode && continue
        (arg isa Expr && arg.head === :(=) && arg.args[1] isa Symbol) ||
            error("@$macroname takes keyword arguments only, like `node = ...`; got `$arg`")
        key = arg.args[1]
        key in allowed ||
            error("@$macroname: unknown keyword `$key`; valid keywords are $(join(("`$k`" for k in allowed), ", "))")
        haskey(keywords, key) && error("@$macroname: keyword `$key` is given twice")
        keywords[key] = arg.args[2]
    end
    for key in required
        haskey(keywords, key) || error("@$macroname: `$key` is required")
    end
    return keywords
end

quoted_symbol(ex) = ex isa QuoteNode && ex.value isa Symbol ? ex.value : nothing

# The members of a cluster as a definition macro writes them: interfaces, `:y`, and members of a
# group by a literal index, `(:T, 1)`. The rule and the dependency macros take the same forms.
function parse_cluster_members(name, keys)
    members = map(keys) do key
        symbol = quoted_symbol(key)
        symbol === nothing || return symbol
        if key isa Expr && key.head === :tuple && length(key.args) == 2 && quoted_symbol(key.args[1]) !== nothing
            key.args[2] isa Integer && return (quoted_symbol(key.args[1]), Int(key.args[2]))
            error("@$name: a group member in a cluster is `(:T, 1)`, with a literal index; got `$key`")
        end
        error("@$name: a cluster member is a symbol like `:y`, or a group member like `(:T, 1)`; got `$key`")
    end
    return Tuple(members)
end

# The argument type rules and traits dispatch on for a node: `Type{<:node}` for a type,
# `typeof(node)` for a function.
node_dispatch_type(node::Type) = Type{<:node}
node_dispatch_type(node) = typeof(node)

instantiate_algorithm(algorithm::Type) = algorithm()
instantiate_algorithm(algorithm) = algorithm
