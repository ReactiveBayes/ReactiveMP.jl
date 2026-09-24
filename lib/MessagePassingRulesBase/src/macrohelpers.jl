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

# The argument type rules and traits dispatch on for a node: `Type{<:node}` for a type,
# `typeof(node)` for a function.
node_dispatch_type(node::Type) = Type{<:node}
node_dispatch_type(node) = typeof(node)

instantiate_algorithm(algorithm::Type) = algorithm()
instantiate_algorithm(algorithm) = algorithm
