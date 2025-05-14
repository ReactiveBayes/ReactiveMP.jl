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
"""
function bottom_type(type)
    @capture(type, (DeltaFn{T_}) | (ReactiveMP.DeltaFn{T_}) | (typeof(T_)) | (Type{<:T_}) | (Type{T_}) | (T_)) ||
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
"""
function upper_type(type)
    if @capture(type, typeof(T_))
        return :(typeof($(ensure_symbol(T))))
    elseif @capture(type, (DeltaFn{T_}) | (ReactiveMP.DeltaFn{T_}))
        return :(Type{<:DeltaFn{<:$T}})
    else
        return :(Type{<:$(bottom_type(type))})
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
    return :($(proxy){<:$(type)})
end

function proxy_type(proxy, type::Expr)
    if @capture(type, Vararg{rest__})
        error("Vararg{T, N} is forbidden in @rule macro, use `ManyOf{N, T}` instead.")
    elseif @capture(type, ManyOf{N_, T_})
        # return :(NTuple{ $N, <: $(proxy_type(proxy, T)) }) # This doesn't work in all of the cases
        return :(ReactiveMP.ManyOf{<:Tuple{Vararg{X, $N} where X <: $(proxy_type(proxy, T))}})
    else
        return :($(proxy){<:$(type)})
    end
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
    @capture(proxy_methods, [methods__]) || error("Invalid specification of proxy methods, should be an array of methods")

    output      = Expr(:block)
    output.args = map(method -> :(($method)(proxy::$(proxy_type)) = ($method)($(proxy_getter)(proxy))), methods)

    return esc(output)
end

__test_inferred_typeof(x)                   = typeof(x)
__test_inferred_typeof(::Type{T}) where {T} = Type{T}

macro test_inferred(T, expression)
    return esc(quote
        let
            local result = Test.@inferred($expression)
            if !(ReactiveMP.MacroHelpers.__test_inferred_typeof(result) <: $T)
                error("Result type $(ReactiveMP.MacroHelpers.__test_inferred_typeof(result)) does not match allowed type $T")
            end
            @test ReactiveMP.MacroHelpers.__test_inferred_typeof(result) <: $T
            result
        end
    end)
end

end
