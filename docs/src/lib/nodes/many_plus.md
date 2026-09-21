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
messages from `UnivariateNormalDistributionsFamily`, including mixed parameterisations and floating
point types.

The node always uses sum-product messages on its edges, independently of the
surrounding factorisation. The forward message sums the incoming means and
variances. A backward message to input `k` subtracts the means of all *other*
inputs from the output mean and adds their variances to the output variance.
It does not depend on the incoming message from input `k` itself.

The node also supports Bethe free-energy scoring using the joint Gaussian
entropy of its inputs. On Gaussian trees its results agree with a chain of
binary additions. In loopy graphs, changing the graph structure and message
schedule can change the inference trajectory.
