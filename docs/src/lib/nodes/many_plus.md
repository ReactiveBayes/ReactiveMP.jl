# [ManyPlus node](@id lib-nodes-many-plus)

```@docs
ManyPlus
```

`ManyPlus` represents `out = sum(inputs)` with a single factor and one edge per
summand. A chain of binary `+` nodes introduces intermediate sum variables. In
loopy models, these variables can require additional initial messages and affect
the message update schedule. `ManyPlus` avoids those intermediate variables, so
initialization can refer directly to the summands.

!!! note "Supported inputs"
    `ManyPlus` currently supports only the summation of **univariate** Gaussian
    variables, optionally mixed with scalar constants. Multivariate Gaussian inputs
    are not supported; use a chain of binary `+` nodes for those instead.

For example, in an RxInfer model:

```julia
@model function sum_model(y, n)
    local x
    for i in 1:n
        x[i] ~ Normal(mean = 0.0, variance = 1.0)
    end
    total := ManyPlus(inputs = x)
    y ~ Normal(mean = total, variance = 0.5)
end
```

The `local x` declaration is required. Without it, `x` exists only inside the
`for` loop and cannot be passed as a whole to `ManyPlus` afterwards.

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
variables. Directly mixing variable references and numeric literals in a vector,
such as `[x1, 2.0, x3]`, is currently not supported by the
[GraphPPL](https://github.com/ReactiveBayes/GraphPPL.jl) model frontend; the
submodel above avoids that limitation.

Constants contribute their value to the sum and zero variance. When every input
message is a point mass, the forward message is also a point mass. The output can
also be observed or fixed, represented by a scalar `PointMass` message. For example:

```julia
@model function observed_sum_model(y, n)
    local x
    for i in 1:n
        x[i] ~ Normal(mean = 0.0, variance = 1.0)
    end
    y ~ ManyPlus(inputs = x)
end

result = infer(
    model = observed_sum_model(n = 3),
    data = (y = 2.0,),
    returnvars = (x = KeepLast(),),
    free_energy = true,
)
```

For a fixed output `y`, the backward message to input `k` has mean
`y - sum(other input means)` and variance `sum(other input variances)`. If all
other inputs are point masses, that backward message is a point mass as well.
Multivariate messages are not supported.

!!! warning "Scaling with the number of inputs"
    Internally, the inputs are stored as a `Tuple`, and the backward message
    to every input depends on a `Tuple` of all the other inputs. Julia compiles
    specialised code for each tuple length, so both compilation and run time grow
    quickly with the number of inputs. In a simple tree model, the first `infer`
    call took about 7 seconds with 50 inputs and about 3 minutes with 200 inputs,
    while later calls took about 0.01 and 0.7 seconds respectively. Every
    iteration also costs ``O(N^2)`` for ``N`` inputs, because each of the ``N``
    backward messages sums over the other ``N - 1`` inputs. `ManyPlus` is
    therefore intended for a moderate number of summands (tens rather than
    hundreds). For very long sums, consider a chain or tree of binary `+` nodes,
    or several `ManyPlus` nodes over blocks of summands.

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

With a fixed output and ``d`` Gaussian inputs, eliminate one Gaussian input using
the sum constraint. The remaining ``d - 1`` dimensions contribute

```math
-H = \frac{\log(\sum_i v_i) - \sum_i \log v_i - (d - 1)(1 + \log 2\pi)}{2},
```

where ``v_i`` are the incoming Gaussian variances. With at most one Gaussian input,
the finite entropy contribution is zero. Scoring retains the constant-input
point-mass counts and adds one for the fixed output.

On Gaussian trees its results agree with a chain of binary additions. In loopy
graphs, changing the graph structure and message schedule can change the inference
trajectory.
