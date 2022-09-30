
# [Math utilities](@id lib-math)

ReactiveMP package exports [`tiny`](@ref) and [`huge`](@ref) objects to represent tiny and huge numbers. These objects aren't really numbers and behave differently depending on the context. They do support any operation that is defined for `Real` numbers. For more info see Julia's documentation about [promotion](https://docs.julialang.org/en/v1/manual/conversion-and-promotion/#Promotion).

```@docs
tiny
huge
```