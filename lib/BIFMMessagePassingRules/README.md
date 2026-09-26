# BIFMMessagePassingRules

Backward-information-filter forward-marginal (BIFM) smoothing of a linear state-space model with
known matrices:

- `BIFM`, a whole time slice, `znext = A zprev + B in` and `out = C znext`, as one deterministic
  node, under `BIFMSmoother(A, B, C)`, which a model must give each BIFM node;
- `BIFMHelper`, at the start of the chain, which turns the backward pass into the forward one.

The rules take multivariate normal messages only. **The free energy of a model with BIFM is not
supported**: asking for it throws a `BIFMFreeEnergyError`.

```julia
using BIFMMessagePassingRules, MessagePassingRulesBase, ExponentialFamily

result = @call_message_update_rule(
    node = BIFM, target = :zprev, algorithm = BIFMSmoother([1.0;;], [1.0;;], [1.0;;]),
    m = (
        out = MvNormalMeanPrecision([1.0], [1.0;;]),
        in = MvNormalMeanPrecision([0.0], [1.0;;]),
        znext = MvNormalWeightedMeanPrecision([0.0], [0.0;;]),
    ),
)
getresult(result)   # the backward message on zprev, N(1, 2) in information form
```

- Documentation: `make docs-bifm` from the repository root builds it into `docs/build`; it will
  be published at <https://reactivebayes.github.io/BIFMMessagePassingRules.jl/dev/>.
- Tests: `make test-bifm`.
- Depends on MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions and
  FastCholesky.
- MIT licence.
