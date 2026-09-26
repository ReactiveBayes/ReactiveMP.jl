# PolyaMessagePassingRules

!!! warning "Licence: GPL-3"
    This package is licensed under **GPL-3**, because its dependency PolyaGammaHybridSamplers is.
    The engine and the other rule packages are MIT and do not depend on it; a model that loads
    this package is subject to the GPL.

Two nodes for count data through the logistic, augmented with Pólya-Gamma variables so that the
messages towards their weights are normal and a normal prior on the weights stays conjugate. Use
them for binomial or logistic regression, and for multinomial regression.

```@docs
PolyaMessagePassingRules
```

## The augmentation

For a count `y` out of `b` trials with log-odds `ψ`, the Pólya-Gamma identity of Polson, Scott
and Windle (2013),

```math
\frac{(e^{\psi})^{y}}{(1 + e^{\psi})^{b}}
  = 2^{-b} e^{\kappa \psi} \int_0^\infty e^{-\omega \psi^2 / 2} \, \mathrm{PG}(\omega \mid b, 0) \, d\omega,
\qquad \kappa = y - b / 2,
```

makes the likelihood Gaussian in `ψ` given `ω ~ PG(b, ψ)`. The rules replace `ω` by its mean at
the current estimate of `ψ`, read from the message on the weights' own edge, which is why a model
must initialise that message. The average energies are those of the original likelihoods, not of
the augmented ones.

## Pages

- [BinomialPolya](@ref page-binomial-polya): `y ~ Binomial(n, σ(xᵀβ))`, with its algorithm.
- [MultinomialPolya](@ref page-multinomial-polya): `x ~ Multinomial(N, p(ψ))` by logistic
  stick-breaking, with its algorithm and helpers.
