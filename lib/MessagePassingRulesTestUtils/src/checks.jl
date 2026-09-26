# Every check is recorded against the user's source line, not a line in this package.
# `description` is a string, or a function producing one, called only when the check fails.
function record_check(passed::Bool, expression, description, source::LineNumberNode)
    text = passed ? nothing : description isa Function ? description() : description
    Test.do_test(Test.Returned(passed, text, source), expression)
    return passed
end

"""
    approximately_equal(a, b; atol, rtol) -> Bool

Whether two rule outputs agree, the comparison every table case, in-place and scratch check
makes. The types must match exactly: a `Normal{Float32}` never equals a `Normal{Float64}`. Then
the values are compared by what they hold:

- numbers by `isapprox`, and arrays, tuples and named tuples (with the same keys) element by
  element;
- a `PointMass` by its location, and any other `Distribution` by its `params`;
- a [`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster) by its blocks,
  which must be the same, and then its components;
- `nothing` equals `nothing`;
- any other immutable struct with fields field by field, such as ExponentialFamily's
  `JointNormal`, which is not a `Distribution`; everything else by `==`.

# Keywords

- `atol`, `rtol`: the absolute and relative tolerances of `isapprox`, both required.

# Examples

```jldoctest; setup = :(using Distributions)
julia> MessagePassingRulesTestUtils.approximately_equal(Normal(0.0, 1.0), Normal(1e-8, 1.0); atol = 1e-6, rtol = 0)
true

julia> MessagePassingRulesTestUtils.approximately_equal(Normal(0.0, 1.0), Normal(0.0f0, 1.0f0); atol = 1e-6, rtol = 0)
false
```
"""
approximately_equal(a, b; atol, rtol) = BayesBase.isequal_typeof(a, b) && values_close(a, b; atol, rtol)

values_close(a::Number, b::Number; atol, rtol) = isapprox(a, b; atol, rtol)
values_close(a::AbstractArray, b::AbstractArray; atol, rtol) =
    size(a) == size(b) && all(((x, y),) -> values_close(x, y; atol, rtol), zip(a, b))
values_close(a::Tuple, b::Tuple; atol, rtol) =
    length(a) == length(b) && all(((x, y),) -> values_close(x, y; atol, rtol), zip(a, b))
values_close(a::NamedTuple, b::NamedTuple; atol, rtol) = keys(a) == keys(b) && values_close(values(a), values(b); atol, rtol)
values_close(a::PointMass, b::PointMass; atol, rtol) = values_close(mean(a), mean(b); atol, rtol)
values_close(a::Distribution, b::Distribution; atol, rtol) = values_close(params(a), params(b); atol, rtol)
values_close(::Nothing, ::Nothing; atol, rtol) = true
values_close(a::FactorizedCluster, b::FactorizedCluster; atol, rtol) =
    cluster_blocks(a) == cluster_blocks(b) && values_close(BayesBase.components(a), BayesBase.components(b); atol, rtol)
values_close(a::T, b::T; atol, rtol) where {T} =
    isstructtype(T) && !ismutabletype(T) && fieldcount(T) > 0 ?
    all(i -> values_close(getfield(a, i), getfield(b, i); atol, rtol), 1:fieldcount(T)) : a == b
values_close(a, b; atol, rtol) = a == b
