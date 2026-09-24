# [Standard rules](@id packages-standard)

```@docs
StandardMessagePassingRules
```

`StandardMessagePassingRules` defines the rules of the standard nodes: distributions,
arithmetic, logic and mixtures. Most of its nodes are types other packages own, such as
ExponentialFamily's and Distributions' distributions and Base's arithmetic functions, so the
package exports only the node types it defines itself; loading it is enough for the engine to
find every rule.

## Nodes

| Group | Nodes |
|---|---|
| Univariate distributions | `NormalMeanVariance`, `NormalMeanPrecision`, `GammaShapeRate`, `Gamma`, `GammaInverse`, `Beta`, `Bernoulli`, `Categorical`, `Dirichlet`, `Poisson`, `Uniform`, [`HalfNormal`](@ref), [`Uninformative`](@ref) |
| Multivariate normals | `MvNormalMeanCovariance`, `MvNormalMeanPrecision`, `MvNormalWeightedMeanPrecision`, `MvNormalMeanScalePrecision`, `MvNormalMeanScaleMatrixPrecision` |
| Matrix and Wishart | `Wishart`, `InverseWishart`, `MatrixNormal`, `MatrixNormalWishart`, `MvNormalGamma`, `MvNormalWishart`, `DirichletCollection` |
| Arithmetic | `+`, `-`, `*`, `dot`, the functions themselves being the nodes |
| Logic | [`AND`](@ref), [`OR`](@ref), [`NOT`](@ref), [`IMPLY`](@ref) |
| Mixtures | [`NormalMixture`](@ref) (also [`GaussianMixture`](@ref)), [`GammaMixture`](@ref), [`Mixture`](@ref) |

## Conventions

- **Variational rules** take the expectations naive variational message passing gives: a
  variance's contribution is `1/E[1/v]` and a covariance's `E[Σ⁻¹]⁻¹`, as the nodes' average
  energies do.
- **Matrix corrections.** `*` and `dot` build precision matrices that may be singular, and
  correct them with `ReplaceZeroDiagonalEntries(tiny)` unless the context sets another
  correction; `NoCorrection()` from MatrixCorrectionTools applies none. MvNormalMeanPrecision
  corrects nothing unless one is set.
- **Sampling.** The `*` rules for two general univariate distributions approximate their
  messages from draws of `ctx.rng`.
- **Products that do not commute.** For a matrix operand, `*` computes `A * in` only.
- **Mixtures.** `NormalMixture` and `GammaMixture` are always variational, under their own
  algorithms. `Mixture` is always belief propagation, under `MixtureBP`; its rules need their
  inputs' log scales, so a graph with one needs log-scale annotations, and it has no average
  energy, so the free energy of a model with one is an error.

```@docs
NormalMixture
GaussianMixture
NormalMixtureVMP
GammaMixture
GammaMixtureVMP
Mixture
MixtureBP
GammaShapeLikelihood
HalfNormal
Uninformative
AND
OR
NOT
IMPLY
```

## Helpers

Small linear-algebra helpers the rules share, and the node packages take qualified
(`StandardMessagePassingRules.mul_trace`). The in-place ones overwrite a dense `Array` and
leave any other argument untouched, so a rule may pass them a view or a number.

```@docs
StandardMessagePassingRules.diageye
StandardMessagePassingRules.negate_inplace!
StandardMessagePassingRules.mul_inplace!
StandardMessagePassingRules.rank1update
StandardMessagePassingRules.mul_trace
```
