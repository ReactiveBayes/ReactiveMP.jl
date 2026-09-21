# [ManyPlus node](@id lib-nodes-many-plus)

```@docs
ManyPlus
```

`ManyPlus` represents `out = sum(inputs)` with a single factor and one edge per
summand. A chain of binary `+` nodes introduces intermediate sum variables. In
loopy models, these variables can require additional initial messages and affect
the message update schedule. `ManyPlus` avoids those intermediate variables, so
initialization can refer directly to the summands.

For example, in an RxInfer model:

```julia
@model function sum_model(y, n)
    for i in 1:n
        x[i] ~ Normal(mean = 0.0, variance = 1.0)
    end
    total := ManyPlus(inputs = x)
    y ~ Normal(mean = total, variance = 0.5)
end
```

At least two inputs are required. The current rules support scalar Gaussian
messages from `UnivariateNormalDistributionsFamily` and scalar constants, represented
by `PointMass` input messages, including mixed parameterisations and numeric types.
For an RxInfer model with fixed inputs:

```julia
@model function sum_three(out, a, b, c)
    out := ManyPlus(inputs = [a, b, c])
end

@model function shifted_sum_model(y, c1, c2)
    x ~ Normal(mean = 0.5, variance = 1.0)
    total ~ sum_three(a = c1, b = x, c = c2)
    y ~ Normal(mean = total, variance = 0.5)
end

result = infer(
    model = shifted_sum_model(),
    data = (y = 2.0, c1 = 2, c2 = -1f0),
    returnvars = (x = KeepLast(), total = KeepLast()),
    free_energy = true,
)
```

This example supplies the fixed summands through `data`. The `sum_three` submodel
assembles their interfaces into one `ManyPlus` factor without intermediate sum
variables. In RxInfer 5.5.2, directly mixing variable references and numeric
literals in a vector such as `[x1, 2.0, x3]` fails during GraphPPL graph construction;
the submodel above avoids that frontend limitation.

Constants contribute their value to the sum and zero variance. When every input
message is a point mass, the forward message is also a point mass. The incoming
message on the output interface must still be Gaussian; directly observed or fixed
outputs and multivariate messages are not supported.

The node always uses sum-product messages on its edges, independently of the
surrounding factorisation. The forward message sums the incoming means and
variances. A backward message to input `k` subtracts the means of all *other*
inputs from the output mean and adds their variances to the output variance.
It does not depend on the incoming message from input `k` itself.

This deterministic node uses one joint local belief over the inputs and output:

```math
q(\mathrm{inputs}, \mathrm{output})
= q(\mathrm{inputs})\,
  \delta\!\left(\mathrm{output} - \sum_i \mathrm{inputs}_i\right).
```

The input belief can contain correlations; the node does not assume a product of
independent input and output marginals. The output is determined exactly by the
inputs, so deterministic-node Bethe free-energy scoring uses the joint input
entropy after eliminating the output. Gaussian input dimensions contribute their
joint Gaussian entropy, while constant inputs retain the point-mass entropy counts
needed for cancellation with the clamped-variable terms.

On Gaussian trees its results agree with a chain of binary additions. In loopy
graphs, changing the graph structure and message schedule can change the inference
trajectory.
