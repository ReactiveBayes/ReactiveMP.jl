# Every check is recorded against the user's source line, not a line in this package.
# `description` is a string, or a function producing one, called only when the check fails.
function record_check(passed::Bool, expression, description, source::LineNumberNode)
    text = passed ? nothing : description isa Function ? description() : description
    Test.do_test(Test.Returned(passed, text, source), expression)
    return passed
end

"""
    approximately_equal(a, b; atol, rtol)

Whether two rule outputs agree: numbers and arrays by `isapprox`, tuples and named tuples
element by element, point masses by their location, other distributions by their
parameters, and any other immutable struct with fields field by field (such as
ExponentialFamily's `JointNormal`, which is not a `Distribution`). Outputs of different types
are never equal.
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
