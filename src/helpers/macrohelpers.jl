module MacroHelpers

using MacroTools

"""
    ReactiveMP.MacroHelpers.@proxy_methods(proxy_type, proxy_getter, proxy_methods)

Define each function of `proxy_methods`, a vector literal, on `proxy_type` as that function of
`proxy_getter` of the value: how [`Message`](@ref ReactiveMP.Message) and [`Marginal`](@ref ReactiveMP.Marginal) forward `mean`, `var` and
the other statistics to their data.

# Examples

```julia
@proxy_methods Message getdata [Distributions.mean, Distributions.var]
```

defines

```julia
Distributions.mean(proxy::Message) = Distributions.mean(getdata(proxy))
Distributions.var(proxy::Message) = Distributions.var(getdata(proxy))
```

# Throws

- `ErrorException` when `proxy_methods` is not a vector literal.
"""
macro proxy_methods(proxy_type, proxy_getter, proxy_methods)
    @capture(proxy_methods, [methods__]) || error(
        "Invalid specification of proxy methods, should be an array of methods",
    )

    output = Expr(:block)
    output.args = map(method -> :(($method)(proxy::$(proxy_type)) = ($method)($(proxy_getter)(proxy))), methods)

    return esc(output)
end

end
