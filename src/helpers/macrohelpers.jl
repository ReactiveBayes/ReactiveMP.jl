module MacroHelpers

using MacroTools

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
Distributions.var(proxy::Message)  = Distributions.var(getdata(proxy))
```
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
