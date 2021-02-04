using MacroTools




module MacroHelpers

using MacroTools

"""
    ensure_symbol(input)

This function ensures that the given argument has a Symbol type
"""
ensure_symbol(input::Symbol) = input
ensure_symbol(input) = error("$(input) is not a symbol.")

"""
    bottom_type(type)

This function returns `T` expression for the following input expressions:
1. typeof(T)
2. Type{ <: T }
3. Type{ T }
4. T

# Arguments
- `type`: Type expression to be lowered

# See also: [`extract_fformtype`](@ref), [`upper_type`](@ref)
"""
function bottom_type(type)
    @capture(type, (typeof(T_)) | (Type{ <: T_ }) | (Type{ T_ }) | (T_)) || 
        error("Expression $(type) doesnt seem to be a valid type expression.")
    return T
end

"""
    upper_type(type)

This function returns `Type{ <: T }` expression for the following input expressions:
1. typeof(T)
2. Type{ <: T }
3. Type{ T }
4. T

# Arguments
- `type`: Type expression to be extended

# See also: [`extract_fformtype`](@ref), [`bottom_type`](@ref)
"""
function upper_type(type)
    if @capture(type, typeof(T_))
        return :(typeof($(ensure_symbol(T))))
    else
        return :(Type{ <: $(bottom_type(type)) })
    end
end

"""
    proxy_type(proxy, type)

Returns a type wrapped with a proxy type in a form of `ProxyType{ <: Type }`.

# Arguments
- `proxy`: Proxy type used to wrap `type`
- `type`: Type to be wrapped
"""
function proxy_type(proxy::Symbol, type::Symbol)
    return :($(proxy){ <: $(type) })
end

function proxy_type(proxy::Symbol, type::Expr)
    if @capture(type, NTuple{N_, T_})
        return :(NTuple{ $N, <: $(proxy_type(proxy, T)) })
    elseif @capture(type, Tuple{ T__ })
        return :(Tuple{ $(map(t -> :( <: $(proxy_type(proxy, t))), T)...) })
    else 
        return :($(proxy){ <: $(type) })
    end
end

end

function __extract_on_args_macro_rule(on)
    if @capture(on, :name_)
        return :(Type{ Val{ $(QuoteNode(name)) } }), nothing
    elseif @capture(on, (:name_, index_))
        return :(Tuple{ Val{$(QuoteNode(name))}, Int}), index
    else
        error("Error in macro. on specification is incorrect")
    end
end

function __extract_fn_args_macro_rule(inputs; specname, prefix, proxy)
    finputs = filter((i) -> startswith(string(first(i)), string(prefix)), inputs)

    names  = map(first, finputs)
    types  = map((i) -> MacroHelpers.proxy_type(proxy, last(i)), finputs)

    @assert all((n) -> length(string(n)) > 2, names)  || error("Empty $(specname) name found in arguments")

    init_block = map(enumerate(names)) do (index, iname)
        return :($(iname) = getdata($(specname)[$(index)]))
    end

    out_names = length(names) === 0 ? :Nothing : :(Type{ Val{ $(tuple(map(n -> Symbol(string(n)[(length(string(prefix)) + 1):end]), names)...)) } })
    out_types = length(types) === 0 ? :Nothing : :(Tuple{ $(types...) })

    return out_names, out_types, init_block
end

function __rearrange_tupled_arguments(name::Symbol, length::Int, swap::Tuple{Int, Int})
    arguments = map(i -> :($(name)[$i]), 1:length)
    tmp = arguments[ first(swap) ]
    arguments[ first(swap) ] = arguments[ last(swap) ]
    arguments[ last(swap) ]  = tmp
    return Expr(:tuple, arguments...)
end