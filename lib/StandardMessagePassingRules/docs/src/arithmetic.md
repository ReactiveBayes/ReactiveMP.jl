# Arithmetic

The arithmetic nodes `+`, `-`, `*` and `dot` are
[deterministic nodes](@extref MessagePassingRulesBase glossary-deterministic-node): their output
is a function of their inputs, such as `out = in1 + in2`. Base's and LinearAlgebra's functions
are the nodes themselves, so a model writes `y ~ x + z` or `y ~ A * x`, and
[`rule_coverage`](@extref MessagePassingRulesBase.rule_coverage)`(+)` takes the function. The
package re-exports LinearAlgebra's `dot` for models to use.

A deterministic node's [clusters](@extref MessagePassingRulesBase glossary-cluster) are always
`out` and the joint over its inputs, whatever the
[factorisation](@extref MessagePassingRulesBase glossary-factorisation). Every rule here is
therefore [belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation) over
[messages](@extref MessagePassingRulesBase glossary-message). The message towards `out` is the
[pushforward](@extref MessagePassingRulesBase glossary-pushforward) of the input messages, the
distribution of `f(in...)`. The message towards an input inverts the function against the other
messages. For normal messages, with one factor of `*` or `dot` known, these messages are exact
and normal. Between two uncertain factors of `*` the result has no closed form, and the rules
return log-densities or samples instead. None of the nodes has an
[average energy](@extref MessagePassingRulesBase glossary-average-energy) of its own.

`+`, `-` and `dot` run under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), and `*` runs under
[`MultiplicationSampling`](@ref), its default. No model names an algorithm for them.

```@setup arithmetic
using MessagePassingRulesBase, StandardMessagePassingRules, ExponentialFamily, Distributions, BayesBase
```

## Example

The sum of a normal and a known value shifts the normal. A known factor of `*` scales the mean,
and it scales the variance by its square:

```jldoctest arithmetic
julia> using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase

julia> sum_message = getresult(@call_message_update_rule(
           node = +, target = :out,
           m = (in1 = NormalMeanVariance(1.0, 1.0), in2 = PointMass(2.0)),
       ));

julia> mean(sum_message) ≈ 3.0 && var(sum_message) ≈ 1.0
true

julia> scaled = getresult(@call_message_update_rule(
           node = *, target = :out,
           m = (A = PointMass(2.0), in = NormalMeanVariance(1.0, 1.0)),
       ));

julia> mean(scaled) ≈ 2.0 && var(scaled) ≈ 4.0
true
```

## Addition, `+`

```math
\mathrm{out} = \mathrm{in}_1 + \mathrm{in}_2
```

| interface | meaning | messages its rules take |
|---|---|---|
| `out` | the sum | a normal or a `PointMass` |
| `in1` | the first term | a normal or a `PointMass` |
| `in2` | the second term | a normal or a `PointMass` |

```@example arithmetic
MessagePassingRulesBase.nodespec(+)
```

```@example arithmetic
MessagePassingRulesBase.rule_coverage(+)
```

The message towards `out` from two normal messages adds their means and their variances:

```@example arithmetic
@call_message_update_rule(
    node = +, target = :out,
    m = (in1 = NormalMeanVariance(1.0, 1.0), in2 = NormalMeanVariance(2.0, 3.0)),
)
```

Towards an input, the rule subtracts. With `out` observed at `5` and a normal message on `in1`,
the message towards `in2` is a normal around `5 - 1`:

```@example arithmetic
@call_message_update_rule(
    node = +, target = :in2,
    m = (out = PointMass(5.0), in1 = NormalMeanVariance(1.0, 1.0)),
)
```

The messages are normal whenever any input is normal, and a `PointMass` when both inputs are
known. A normal shifted by a known value keeps its mean–precision or weighted-mean form. Any
other result is in mean–variance or mean–covariance form. Towards `out`, Distributions'
`convolve` sums any other two distributions, and it throws a `MethodError` for a pair it does not
know. The joint marginal of the inputs is a joint normal when both inputs are normal, and
factorised when one is known. Every rule has [log scale](@extref MessagePassingRulesBase glossary-log-scale)
zero.

## Subtraction, `-`

```math
\mathrm{out} = \mathrm{in}_1 - \mathrm{in}_2
```

| interface | meaning | messages its rules take |
|---|---|---|
| `out` | the difference | a normal or a `PointMass` |
| `in1` | the minuend | a normal or a `PointMass` |
| `in2` | the subtrahend | a normal or a `PointMass` |

```@example arithmetic
MessagePassingRulesBase.rule_coverage(-)
```

Its rules are `+`'s, rearranged. Towards `out`, the message is the difference of the inputs.
Towards `in1`, it is the sum of `out` and `in2`, and it takes any two distributions through
`convolve`, as `+` does towards `out`. Towards `in2`, it is the difference of `in1` and `out`.

## Multiplication, `*`

```math
\mathrm{out} = A \cdot \mathrm{in}
```

| interface | meaning | messages its rules take |
|---|---|---|
| `out` | the product | a normal, a Gamma, a `PointMass`, or any univariate distribution |
| `A` | the left factor | a `PointMass` (scalar, vector, matrix or `UniformScaling`), a univariate normal, or any univariate distribution |
| `in` | the right factor | a `PointMass`, a normal, a Gamma, or any univariate distribution |

```@example arithmetic
MessagePassingRulesBase.nodespec(*)
```

```@example arithmetic
MessagePassingRulesBase.rule_coverage(*)
```

With a known matrix `A`, the message towards `out` maps the normal message on `in` through `A`:
the mean becomes `A μ` and the covariance `A Σ Aᵀ`:

```@example arithmetic
@call_message_update_rule(
    node = *, target = :out,
    m = (A = PointMass([1.0 1.0; 0.0 1.0]), in = MvNormalMeanCovariance([1.0, 2.0], diageye(2))),
)
```

Towards `in`, the message is a normal in weighted-mean form, with precision `Aᵀ Σ⁻¹ A` and
weighted mean `Aᵀ Σ⁻¹ m` from the message `N(m, Σ)` on `out`. The card lists the
[service](@extref MessagePassingRulesBase glossary-service) the rule reads,
[`matrix_correction`](@extref MessagePassingRulesBase.matrix_correction):

```@example arithmetic
@call_message_update_rule(
    node = *, target = :in,
    m = (out = MvNormalMeanCovariance([3.0, 2.0], diageye(2)), A = PointMass([1.0 1.0; 0.0 1.0])),
)
```

With one factor known, the rules are in closed form. Towards `out`, the message is a scaled
normal or Gamma. Towards the unknown factor, it is a normal in weighted-mean form or a Gamma, with
the log scale `-d log |c|` of a known scalar `c`. A known factor may be a scalar, a vector or a
matrix. With a vector, the other factor is a scalar. With a matrix, the rules compute only
`A * in`, never `in * A`, so they refuse a matrix `in` towards `A`. The precision matrices that
the rules towards an input build may be singular. The rules correct them with the context's
`matrix_correction`, `ReplaceZeroDiagonalEntries(tiny)` by default.

Between two univariate normals, the messages are log-densities, a `ContinuousUnivariateLogPdf`.
Towards an input, the rule computes the exact integral. Towards `out`, it sums a Bessel series
truncated at ten terms. Between two other univariate distributions, the rules draw samples from
the context's `rng`, as many as the algorithm says. These messages are unnormalised and declare
no log scale.

```@docs
MultiplicationSampling
```

## Dot product, `dot`

```math
\mathrm{out} = \mathrm{in}_1^\top \mathrm{in}_2
```

| interface | meaning | messages its rules take |
|---|---|---|
| `out` | the scalar product | a univariate normal |
| `in1` | the first vector | a `PointMass` or a normal |
| `in2` | the second vector | a `PointMass` or a normal |

```@example arithmetic
MessagePassingRulesBase.rule_coverage(dot)
```

**Limitations.** One input must be known. Two normal inputs have no closed form. The rules for
that case exist only to throw an error that points to the `SoftDot` node, whose package handles
it. The precision `a w aᵀ` towards an input has rank one, and the rule corrects it with the
context's [`matrix_correction`](@extref MessagePassingRulesBase.matrix_correction),
`ReplaceZeroDiagonalEntries(tiny)` by default.
