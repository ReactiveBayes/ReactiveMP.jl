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
a model initialises that message. The node declares none, since the dimension of the
weights is the model's.

```@docs
BinomialPolya
BinomialPolyaApproximation
MultinomialPolya
MultinomialPolyaApproximation
logistic_stick_breaking
compose_Nks
```

BinomialPolya's average energy takes the expectation of `softplus(xᵀβ)` under `q(β)` by
Gauss–Hermite cubature. MultinomialPolya's includes the multinomial coefficient's
expectation, `log N! - Σ ⟨log x_k!⟩`, for a Multinomial `q(x)` of `N` trials.
