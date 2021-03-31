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
function proxy_type(proxy::Symbol, type)
    if @capture(type, NTuple{N_, T_})
        return :(NTuple{ $N, <: $(proxy_type(proxy, T)) })
    elseif @capture(type, AbstractVector{T_})
        return :(AbstractVector{ <: $(proxy_type(proxy, T)) })
    elseif @capture(type, AbstractVector)
        return :(AbstractVector{ <: $proxy })
    elseif @capture(type, Tuple{ T__ })
        return :(Tuple{ $(map(t -> :( <: $(proxy_type(proxy, t))), T)...) })
    else 
        return :($(proxy){ <: $(type) })
    end
end

"""
    rearranged_tuple(name::Symbol, length::Int, swap::Tuple{Int, Int})
"""
function rearranged_tuple(name::Symbol, length::Int, swap::Tuple{Int, Int})
    args = map(i -> :($(name)[$i]), 1:length)
    i, j = first(swap), last(swap)
    args[i], args[j] = args[j], args[i]
    return Expr(:tuple, args...)
end

end

