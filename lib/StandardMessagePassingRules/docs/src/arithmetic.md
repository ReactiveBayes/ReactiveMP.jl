# Arithmetic

The deterministic nodes `+`, `-`, `*` and `dot`: Base's and LinearAlgebra's functions are the
nodes themselves, so a model writes `y ~ x + z` or `y ~ A * x`, and
`MessagePassingRulesBase.rule_coverage(+)` takes the function. The package re-exports
LinearAlgebra's `dot` for models to use. A deterministic node's clusters are always `out` and
the joint over its inputs, so every rule here is belief propagation over messages, whatever the
factorisation, and none has an average energy of its own.

`+`, `-` and `dot` run under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), `*` under
[`MultiplicationSampling`](@ref), its default, so no model names an algorithm for them.

```@setup arithmetic
using MessagePassingRulesBase, StandardMessagePassingRules, ExponentialFamily, Distributions
```

## Example

The sum of a normal and a known value shifts the normal; the message towards an input of `+`
subtracts:

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
MessagePassingRulesBase.rule_coverage(+)
```

The messages are normal whenever any input is, and a `PointMass` when both are known; a normal
shifted by a known value keeps a mean–precision or weighted-mean form, and any other result is
in mean–variance or mean–covariance form. Towards `out`, any other two distributions are summed
by Distributions' `convolve`, which is a `MethodError` for a pair it does not know. The joint marginal of the inputs is a joint normal when both are normal,
and factorised when one is known. Every rule has log scale zero.

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

Its rules are `+`'s, rearranged: towards `out` the difference, towards `in1` the sum of `out` and
`in2`, which takes any two distributions through `convolve` as `+` does towards `out`, and
towards `in2` the difference of `in1` and `out`.

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
MessagePassingRulesBase.rule_coverage(*)
```

With one factor known the rules are closed-form: a scaled normal or Gamma forwards, and
backwards a normal in weighted-mean form or a Gamma, with the log scale `-d log |c|` of a known
scalar `c`. A known factor may be a scalar, a vector (then the other factor is a scalar) or a
matrix; for a matrix only `A * in` is computed, never `in * A`, so a matrix `in` towards `A` is
refused. Precision matrices built backwards may be singular, and are corrected with the context's
`matrix_correction`, `ReplaceZeroDiagonalEntries(tiny)` by default.

Between two univariate normals the messages are log-densities (a `ContinuousUnivariateLogPdf`):
towards an input the exact integral, towards `out` a Bessel series truncated at ten terms.
Between two other univariate distributions they are sampled, from the context's `rng`, with as
many draws as the algorithm says; they are unnormalised and declare no log scale.

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

**Limitations.** One input must be known. Two normal inputs have no closed form: the rules for
them exist, so that the call fails with a message, and throw an error that points to the
`SoftDot` node, whose package handles that case. The backward precision `a w aᵀ` has rank one
and is corrected with the context's `matrix_correction`, `ReplaceZeroDiagonalEntries(tiny)` by
default.
