# RxInferExamples on v6 and v7

Five models from [RxInferExamples.jl](https://github.com/ReactiveBayes/RxInferExamples.jl),
run on **RxInfer 5.5.2 over ReactiveMP 6.5.0** (v6) and on **RxInfer's `refactor/reactivemp-v7`
branch over this checkout** (v7), with their posteriors and free energies compared.

The notebooks are read from a sibling checkout, `../../../RxInferExamples.jl`, and only their model
and inference code is taken: plotting is dropped, and data loading uses plain Julia.

## Running

From the repository root:

```bash
# once: resolve the two environments (Julia 1.13)
julia --project=compat/rxinfer-examples -e 'import Pkg; Pkg.instantiate()'
julia --project=compat/rxinfer-examples/v6-extras -e 'import Pkg; Pkg.instantiate()'

# each example, on both sides; each writes results/<name>_<side>.jls
for e in kalman delta hmm gmm ar; do
    compat/rxinfer-examples/run_v6.sh $e
    compat/rxinfer-examples/run_v7.sh $e
done

# compare, plain Julia; exit code 1 if any entry is over the tolerance
julia compat/rxinfer-examples/compare.jl            # all, atol = 1e-8
julia compat/rxinfer-examples/compare.jl ar hmm     # some
ATOL=1e-6 julia compat/rxinfer-examples/compare.jl  # another tolerance
```

- `run_v6.sh` runs `<name>_v6.jl` in `../v6-comparison` (ReactiveMP 6.5.0 and RxInfer 5.5.2,
  pinned), unmodified. The one registry package the examples need that it lacks, StableRNGs, comes
  from `v6-extras/`, stacked after it on `JULIA_LOAD_PATH`.
- `run_v7.sh` runs `<name>_v7.jl` in this directory's `Project.toml`: RxInfer from
  `../../../RxInfer.jl`, the rule packages from `../../lib`.
- Each script prints which RxInfer and ReactiveMP it loaded, and from where, so a run that picked up
  the wrong engine shows.
- `common.jl` holds the notebooks' data generators, copied verbatim and shared by both sides so
  the data is identical (it is recorded too, as `data:*`, and compared), and the recorder.
- `results/` is not meant to be committed (`.gitignore`); it is about 7 MB.

## What is recorded

For every posterior, per index and per iteration where the notebook keeps them: the mean and
variance (or covariance) of a Gaussian, whatever its parameterisation; shape and rate of a Gamma;
the parameters of a Beta, Dirichlet or DirichletCollection; the probabilities of a Categorical;
the mean and variance of a Wishart. The free-energy trace wherever RxInfer computes one. An entry
agrees when every value is within `atol = 1e-8` of v6's.

## The examples

| name | notebook | models | v7 port |
|---|---|---|---|
| `kalman` | Basic Examples/Kalman filtering and smoothing | `rotate_ssm`, a 2-D linear Gaussian SSM, exact (free energy); `smoothing`, a random walk with missing observations and a Gamma prior on the noise precision, VMP | none |
| `delta` | the same notebook, its nonlinear parts | `identification_problem` with `s := x + w` (addition node) and with `s := smooth_min(x, w)` (**Delta**, Linearization), structured VMP with Gamma priors, 50 iterations; `rx_identification`, the streaming version with `@autoupdates`, 300 steps × 10 iterations | `@meta`/`meta` → `@algorithm`/`algorithm`, `Linearization()` → `DeltaApproximation(method = Linearization())` |
| `hmm` | Basic Examples/Hidden Markov Model | 3-state HMM, `DiscreteTransition` with DirichletCollection priors, `q(s_0, s)q(A)q(B)`, 20 iterations, free energy | `using DiscreteTransitionMessagePassingRules` |
| `gmm` | Problem Specific/Gaussian Mixture | univariate 2-component `NormalMixture` (Beta, Normal, Gamma priors); bivariate 6-component one (MvNormal, Wishart, Dirichlet); mean-field, free energy | none |
| `ar` | Problem Specific/Autoregressive Models | latent AR(5) (free energy); sinusoidal AR(2) with 100 predicted points; AAL stock AR(50) with 50 predicted points; ARMA(10, 4) on the stock data | `using AutoregressiveMessagePassingRules`, `ARMeta` → `ARVMP`, `ReactiveMP.ar_unit` → `AutoregressiveMessagePassingRules.ar_unit` |

Departures from the notebooks, the same on both sides:

- `kalman`'s missing-data part draws its noise from `StableRNG(42)`; the notebook uses the global
  generator, which is not reproducible. It also keeps `τ`, which the notebook does not return.
- `delta`'s two batch identification runs compute the free energy, which the notebook does not.
- `ar` reads `aal_stock.csv` with `readdlm` instead of CSV.jl/DataFrames.jl (the file has no missing
  entries), and keeps only the means and variances of the AR(50)'s states, not their 50×50
  covariances.

## Results

Julia 1.13.0, 2026-09-25.

| example | entries | max \|Δ\| | free energy, v6 = v7 (final) | result |
|---|---|---|---|---|
| `kalman` | 573 | 0 (bit-identical) | 1891.64719346 | pass |
| `delta` | 3456 | 0 (bit-identical, streaming included) | + : 852.845929625; smooth_min: 383.492446809; streaming: 1.64885144788 | pass |
| `hmm` | 104 | 7.1e-15 (A), 9.1e-13 (free energy) | 64.585164497 | pass |
| `gmm` | 1304 | 7.3e-12 (free energy); posteriors 0 | univariate 138.184486211; multivariate 3884.36927091 | pass |
| `ar` | 7673 | 0 (bit-identical, ARMA included) | AR(5): 994.91342847 | pass |

No declared correction is involved in any of these models, and none was needed to explain a
difference: every Gamma, Wishart and NormalMeanVariance node here has point-mass parameters, where
the corrections (#672's Gamma/GammaInverse energies, #675's Wishart rule, the `1/E[1/v]` variational
rules) give v6's values.

### `ar`: bit-identical, once `dot` and `+` round as v6 did

The first run differed on the sinusoidal AR(2) by up to 9.7e-6 (relative 1.5e-6). The cause was not
a formula but two representation changes in Standard's ports, which this comparison found and which
are now fixed:

1. **`dot`'s precision towards an input** was `a * w * a'`, where v6 built `v_a_vT(a, w)`: for a
   dense `a`, `(a aᵀ) w`, exactly symmetric where `(a w) aᵀ` is not always (the ARMA's 1209 `dot`
   messages made FastCholesky warn "not symmetric" 49 times); and for the AR package's standard basis
   vector (the model's `c = ar_unit(...)`), a `Diagonal`, where the port built a dense matrix. At the
   last observed point that message reaches AR's joint `q(y, x)` alone, whose `mean_cov` and `cholinv`
   round differently for a dense matrix, by 1.1e-16 in the first iteration; the model is
   ill-conditioned (poles on the unit circle, `γ = 0.01`, 100 steps unobserved), and the difference
   doubled roughly every iteration. Standard has `v_a_vT` again, with the AR package's method for its
   basis vector.
2. **`+`'s messages** took a multivariate normal's `mean` and `cov` separately where v6 took
   `mean_cov`, which for an `MvNormalWeightedMeanPrecision` round 1 ulp apart; they take `mean_cov`.

Found by tracing every rule call, product and marginal on both sides through RxInfer's `callbacks`:
all were bit-identical up to the 350th AR `θ` message of the first iteration.
