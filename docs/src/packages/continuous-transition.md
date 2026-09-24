# [ContinuousTransition](@id packages-continuous-transition)

```@docs
ContinuousTransitionMessagePassingRules
```

`ContinuousTransition` is `y ~ N(f(a) x, W⁻¹)`: `x` is carried to `y` through a matrix built
from the vector `a` by the transformation of its algorithm, `CTVMP(f)`, which the model must give.
Its rules are variational, under mean-field or the structured `q(y, x)`. Its rule towards `a` also
reads `q(a)`, the point it expands `f` around, which its declaration adds to the default scheme's
inputs ([Extending the default scheme](@ref rules-algorithms-extending)). v6 called the algorithm
`CTMeta`.

```@docs
ContinuousTransition
CTVMP
```

!!! note
    v6's average energies were wrong in three terms: `⟨log det W⟩` was not halved, the dimension
    was taken as half of `q(y)`'s or of `q(y, x)`'s, and the identity stood in for `x`'s
    covariance beside the uncertainty of `a`. They are the closed form now, which a Monte Carlo
    estimate confirms.

!!! note
    Every rule linearises `f` the same way. v6's rules towards `a` and `W` took the rows of
    `f(a)` as linear in `a` through the origin, and so dropped the offset `f(m_a) - J m_a` of an
    affine or nonlinear `f`, such as a rotation, which its other rules kept. Nothing changes for
    `f = a -> reshape(a, n, m)`.
