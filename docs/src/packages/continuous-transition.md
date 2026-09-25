# [ContinuousTransition](@id packages-continuous-transition)

```@docs
ContinuousTransitionMessagePassingRules
```

`ContinuousTransition` is `y ~ N(f(a) x, W⁻¹)`: `x` is carried to `y` through a matrix built
from the vector `a` by the transformation of its algorithm, `CTVMP(f)`, which the model must give.
Its rules are variational, under mean-field or the structured `q(y, x)`. Its rule towards `a` also
reads `q(a)`, the point it expands `f` around, which its declaration adds to the default scheme's
inputs ([Extending the default scheme](@ref rules-algorithms-extending)).

```@docs
ContinuousTransition
CTVMP
```

The average energies are the closed form, which the package's tests confirm against a Monte
Carlo estimate. Every rule linearises `f` the same way, around the mean of `q(a)`, keeping the
offset `f(m_a) - J m_a` of an affine or nonlinear `f`, such as a rotation; for
`f = a -> reshape(a, n, m)` the offset is zero.
