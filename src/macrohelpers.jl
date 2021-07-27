module MacroHelpers

using MacroTools: postwalk
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
function proxy_type(proxy, type::Symbol)
    return :($(proxy){ <: $(type) })
end

function proxy_type(proxy, type::Expr)
    if @capture(type, NTuple{N_, T_})
        return :(NTuple{ $N, <: $(proxy_type(proxy, T)) })
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

"""
    @proxy_methods(proxy_type, proxy_getter, proxy_methods)

Generates proxy methods for a specified `proxy_type` using `proxy_getter`. For example:
```julia
@proxy_methods Message getdata [ 
    Distributions.mean, 
    Distributions.var 
]
```

generates:
```julia
Distributions.mean(proxy::Message) = Distributions.mean(getdata(proxy))
Distributions.var(proxy::Message)  = Distributions.mean(getdata(proxy))
```
"""
macro proxy_methods(proxy_type, proxy_getter, proxy_methods)
    
    @capture(proxy_methods, [ methods__ ]) || error("Invalid specification of proxy methods, should be an array of methods")
    
    output      = Expr(:block)
    output.args = map(method -> :(($method)(proxy::$(proxy_type)) = ($method)($(proxy_getter)(proxy))), methods)
    
    return esc(output)
end

function expression_convert_eltype(eltype::Type{T}, expr::Expr) where T
    if @capture(expr, f_(args__)) 
        return :(ReactiveMP.convert_eltype($f, $T, $expr))
    elseif @capture(expr, (elems__, ))
        entries = map(elems) do elem
            @capture(elem, (name_ = value_)) || error("Invalid expression specification in expression_convert_eltype() function: $expr. Expression should be in the form of a constructor call or tuple of (name = value) elements.")
            return (name, value)
        end
        return Expr(:tuple, map((entry) -> :($(first(entry)) = $(expression_convert_eltype(eltype, last(entry)))), entries)...)
    end
    error("Invalid expression specification in expression_convert_eltype() function: $expr. Expression should be in the form of a constructor call or tuple of (name = value) elements.")
end

end

