"""
    public_equivalent(d)

`d` as the type users expect: the same distribution, converted from an efficient working type
to its public counterpart. The identity by default.

Rules may compute in types chosen for the arithmetic rather than for the reader.
ExponentialFamily's `WishartFast`, for instance, stores the inverse of its scale matrix, which
is what the Wishart rules and products need, while users and most code expect Distributions'
`Wishart`. A package whose rules return such a working type adds a method for it:

```julia
MessagePassingRulesBase.public_equivalent(d::WishartFast) = convert(Wishart, d)
```

An engine applies it to every marginal it forms from messages, so a posterior reaches its
reader, and the rules that consume that marginal, in the public type. A method must return
the same distribution, up to rounding; it is a change of representation, never of meaning.
A working type that no reader should see is its use case; a type that is public already needs
no method.
"""
public_equivalent(d) = d
