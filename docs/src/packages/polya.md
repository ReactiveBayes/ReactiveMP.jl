# [Pólya](@id packages-polya)

```@docs
PolyaMessagePassingRules
```

!!! warning "License"
    This package depends on PolyaGammaHybridSamplers, which is GPL-3 licensed, and so is GPL-3
    itself. The engine and the other rule packages do not depend on it.

`BinomialPolya` is a binomial regression through the logistic and `MultinomialPolya` a
multinomial one through logistic stick-breaking. Pólya-Gamma augmentation makes the messages
towards their weights normal. The rules towards the weights read the message on their own edge,
through their declarations ([Extending the default scheme](@ref rules-algorithms-extending)), so
a model initialises that message, as in v6. The node declares none, since the dimension of the
weights is the model's.

```@docs
BinomialPolya
BinomialPolyaApproximation
MultinomialPolya
MultinomialPolyaApproximation
logistic_stick_breaking
compose_Nks
```

!!! note
    v6's average energies were wrong. BinomialPolya's took `softplus` at the mean of `xᵀβ`, even
    when asked to sample, and is now its expectation under `q(β)`, by Gauss–Hermite cubature.
    MultinomialPolya's, for a Multinomial `q(x)` with more than one trial, flipped the sign of
    `Σ ⟨log x_k!⟩` and took `log N` for `log N!`; observed counts were right.
