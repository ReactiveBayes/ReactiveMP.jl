# [API Tutorial](@id examples-api-tutorial)

```julia
# Reactive programming package for Julia
using Rocket 
# Core package for Constrained Bethe Free Energy minimsation with Factor graphs and message passing
using ReactiveMP 
# High-level user friendly probabilistic model and constraints specification language package for ReactiveMP
using GraphPPL
# Optionally include the Distributions.jl package and the Random package from Base
using Distributions, Random
```

    ┌ Info: Precompiling ReactiveMP [a194aa59-28ba-4574-a09c-4a745416d6e3]
    └ @ Base loading.jl:1423
    ┌ Info: Precompiling GraphPPL [b3f8163a-e979-4e85-b43e-1f63d8c8b42c]
    └ @ Base loading.jl:1423


This notebook covers the fundamentals of the ReactiveMP.jl package. For a more advanced usage we refer the interested reader to the documentation.

This tutorial is also available in the [documentation](https://biaslab.github.io/ReactiveMP.jl/stable/).

## General model specification syntax

We use the `@model` macro from the `GraphPPL.jl` package to create a probabilistic model $p(s, y)$ and we also specify extra constraints on the variational family of distributions $\mathcal{Q}$, used for approximating intractable posterior distributions.
Below there is a simple example of the general syntax for model specification. In this tutorial we do not cover all possible ways to create models or advanced features of `GraphPPL.jl`.  Instead we refer the interested reader to the documentation for a more rigorous explanation and illustrative examples.


```julia
# the `@model` macro accepts a regular Julia function
@model function test_model1(s_mean, s_precision)
    
    # We use the `randomvar` function to create 
    # a random variable in our model
    s = randomvar()
    
    # the `tilde` operator creates a functional dependency
    # between variables in our model and can be read as 
    # `sampled from` or `is modeled by`
    s ~ GaussianMeanPrecision(s_mean, s_precision)
    
    # We use the `datavar` function to create 
    # observed data variables in our models
    # We also need to specify the type of our data 
    # In this example it is `Float64`
    y = datavar(Float64)
    
    y ~ GaussianMeanPrecision(s, 1.0)
    
    # In general `@model` macro returns a variable of interests
    # However it is also possible to obtain all variable in the model 
    # with the `ReactiveMP.getvardict(model)` function call
    return s, y 
end
```

The `@model` macro creates a function with the same name and with the same set of input arguments as the original function (`test_model1(s_mean, s_precision)` in this example). However, the return value is modified in such a way to contain a reference to the model object as the first value and to the user specified variables in the form of a tuple as the second value.


```julia
model, (s, y) = test_model1(0.0, 1.0)
```




    (FactorGraphModel(), (RandomVariable(s), DataVariable(y)))



Another way of creating the model is to use the `Model` function that returns an instance of `ModelGenerator`:


```julia
modelgenerator = Model(test_model1, 0.0, 1.0)

model, (s, y) = ReactiveMP.create_model(modelgenerator)
```




    (FactorGraphModel(), (RandomVariable(s), DataVariable(y)))



The benefits of using model generator as a way to create a model is that it allows to change inference constraints and meta specification for nodes. We will talk about factorisation and form constraints and meta specification later on in this demo.

`GraphPPL.jl` returns a factor graph-based representation of a model. We can examine this factor graph structure with the help of some utility functions such as: 
- `getnodes()`: returns an array of factor nodes in a correposning factor graph
- `getrandom()`: returns an array of random variable in the model
- `getdata()`: returns an array of data inputs in the model
- `getconstant()`: return an array of constant values in the model


```julia
getnodes(model)
```




    2-element Vector{ReactiveMP.AbstractFactorNode}:
     FactorNode:
     form            : NormalMeanPrecision
     sdtype          : Stochastic()
     interfaces      : (Interface(out, Marginalisation()), Interface(μ, Marginalisation()), Interface(τ, Marginalisation()))
     factorisation   : ((1,), (2,), (3,))
     local marginals : (:out, :μ, :τ)
     metadata        : nothing
     pipeline        : FactorNodePipeline(functional_dependencies = DefaultFunctionalDependencies(), extra_stages = EmptyPipelineStage()
    
     FactorNode:
     form            : NormalMeanPrecision
     sdtype          : Stochastic()
     interfaces      : (Interface(out, Marginalisation()), Interface(μ, Marginalisation()), Interface(τ, Marginalisation()))
     factorisation   : ((1,), (2,), (3,))
     local marginals : (:out, :μ, :τ)
     metadata        : nothing
     pipeline        : FactorNodePipeline(functional_dependencies = DefaultFunctionalDependencies(), extra_stages = EmptyPipelineStage()





```julia
getrandom(model) .|> name
```




    1-element Vector{Symbol}:
     :s




```julia
getdata(model) .|> name
```




    1-element Vector{Symbol}:
     :y




```julia
getconstant(model) .|> getconst
```




    3-element Vector{Float64}:
     0.0
     1.0
     1.0



It is also possible to use control flow statements such as `if` or `for` blocks in the model specification function. In general, any valid snippet of Julia code can be used inside the `@model` block. As an example consider the following (valid!) model:


```julia
@model function test_model2(n)
    
    if n <= 1
        error("`n` argument must be greater than one.")
    end
    
    # `randomvar(n)` creates a dense sequence of 
    # random variables
    s = randomvar(n)
    
    # `datavar(Float64, n)` creates a dense sequence of 
    # observed data variables of type `Float64`
    y = datavar(Float64, n)
    
    s[1] ~ GaussianMeanPrecision(0.0, 0.1)
    y[1] ~ GaussianMeanPrecision(s[1], 1.0)
    
    for i in 2:n
        s[i] ~ GaussianMeanPrecision(s[i - 1], 1.0)
        y[i] ~ GaussianMeanPrecision(s[i], 1.0)
    end
    
    return s, y
end
```

There are some limitations though regarding using `if`-blocks to create random variables. It is advised to create random variables in advance before `if` block, e.g instead of 

```julia
if some_condition
    x ~ Normal(0.0, 1.0)
else
    x ~ Normal(0.0, 100.0)
end
```

some needs to write:

```julia
x = randomvar()

if some_condition
    x ~ Normal(0.0, 1.0)
else
    x ~ Normal(0.0, 100.0)
end
```



```julia
model, (s, y) = test_model2(10);
```


```julia
# An amount of factor nodes in generated Factor Graph
getnodes(model) |> length
```




    20




```julia
# An amount of random variables
getrandom(model) |> length
```




    10




```julia
# An amount of data inputs
getdata(model) |> length
```




    10




```julia
# An amount of constant values
getconstant(model) |> length
```




    21



It is also possible to use complex expression inside the functional dependency expressions

```julia
y ~ NormalMeanPrecision(2.0 * (s + 1.0), 1.0)
```

The `~` operator automatically creates a random variable if none was created before with the same name and throws an error if this name already exists

```julia
# s = randomvar() here is optional
# `~` creates random variables automatically
s ~ NormalMeanPrecision(0.0, 1.0)
```

An example model which will throw an error:


```julia
@model function error_model1()
    s = 1.0
    s ~ NormalMeanPrecision(0.0, 1.0)
end
```


    LoadError: Invalid name 's' for new random variable. 's' was already initialized with '=' operator before.
    in expression starting at /Users/bvdmitri/.julia/dev/GraphPPL.jl/src/model.jl:173

    

    Stacktrace:

      [1] error(s::String)

        @ Base ./error.jl:33

      [2] (::GraphPPL.var"#39#52"{ReactiveMPBackend, Set{Symbol}, Set{Symbol}, Symbol})(expression::Expr)

        @ GraphPPL ~/.julia/dev/GraphPPL.jl/src/model.jl:320

      [3] walk

        @ ~/.julia/packages/MacroTools/PP9IQ/src/utils.jl:112 [inlined]

      [4] postwalk

        @ ~/.julia/packages/MacroTools/PP9IQ/src/utils.jl:122 [inlined]

      [5] (::MacroTools.var"#21#22"{GraphPPL.var"#39#52"{ReactiveMPBackend, Set{Symbol}, Set{Symbol}, Symbol}})(x::Expr)

        @ MacroTools ~/.julia/packages/MacroTools/PP9IQ/src/utils.jl:122

      [6] iterate

        @ ./generator.jl:47 [inlined]

      [7] collect_to!(dest::Vector{Any}, itr::Base.Generator{Vector{Any}, MacroTools.var"#21#22"{GraphPPL.var"#39#52"{ReactiveMPBackend, Set{Symbol}, Set{Symbol}, Symbol}}}, offs::Int64, st::Int64)

        @ Base ./array.jl:782

      [8] collect_to!(dest::Vector{LineNumberNode}, itr::Base.Generator{Vector{Any}, MacroTools.var"#21#22"{GraphPPL.var"#39#52"{ReactiveMPBackend, Set{Symbol}, Set{Symbol}, Symbol}}}, offs::Int64, st::Int64)

        @ Base ./array.jl:790

      [9] collect_to_with_first!(dest::Vector{LineNumberNode}, v1::LineNumberNode, itr::Base.Generator{Vector{Any}, MacroTools.var"#21#22"{GraphPPL.var"#39#52"{ReactiveMPBackend, Set{Symbol}, Set{Symbol}, Symbol}}}, st::Int64)

        @ Base ./array.jl:760

     [10] _collect(c::Vector{Any}, itr::Base.Generator{Vector{Any}, MacroTools.var"#21#22"{GraphPPL.var"#39#52"{ReactiveMPBackend, Set{Symbol}, Set{Symbol}, Symbol}}}, #unused#::Base.EltypeUnknown, isz::Base.HasShape{1})

        @ Base ./array.jl:754

     [11] collect_similar(cont::Vector{Any}, itr::Base.Generator{Vector{Any}, MacroTools.var"#21#22"{GraphPPL.var"#39#52"{ReactiveMPBackend, Set{Symbol}, Set{Symbol}, Symbol}}})

        @ Base ./array.jl:653

     [12] map(f::Function, A::Vector{Any})

        @ Base ./abstractarray.jl:2849

     [13] walk(x::Expr, inner::Function, outer::GraphPPL.var"#39#52"{ReactiveMPBackend, Set{Symbol}, Set{Symbol}, Symbol})

        @ MacroTools ~/.julia/packages/MacroTools/PP9IQ/src/utils.jl:112

     [14] postwalk(f::Function, x::Expr)

        @ MacroTools ~/.julia/packages/MacroTools/PP9IQ/src/utils.jl:122

     [15] generate_model_expression(backend::ReactiveMPBackend, model_options::Expr, model_specification::Expr)

        @ GraphPPL ~/.julia/dev/GraphPPL.jl/src/model.jl:287

     [16] var"@model"(__source__::LineNumberNode, __module__::Module, model_options::Any, model_specification::Any)

        @ GraphPPL ~/.julia/dev/GraphPPL.jl/src/model.jl:177

     [17] eval

        @ ./boot.jl:373 [inlined]

     [18] include_string(mapexpr::typeof(REPL.softscope), mod::Module, code::String, filename::String)

        @ Base ./loading.jl:1196


By default the `GraphPPL.jl` package creates new references for constants (literals like `0.0` or `1.0`) in a model. In some situations this may not be efficient, especially when these constants represent large matrices. `GraphPPL.jl` will by default create new copies of some constant (e.g. matrix) in a model every time it uses it. However it is possible to use `constvar()` function to create and reuse similar constants in the model specification syntax as

```julia
# Creates constant reference in a model with a prespecified value
c = constvar(0.0)
```

An example:


```julia
@model function test_model5(dim::Int, n::Int, A::Matrix, P::Matrix, Q::Matrix)
    
    s = randomvar(n)
    
    y = datavar(Vector{Float64}, n)
    
    # Here we create constant references
    # for constant matrices in our model 
    # to make inference more memory efficient
    cA = constvar(A)
    cP = constvar(P)
    cQ = constvar(Q)
    
    s[1] ~ MvGaussianMeanCovariance(zeros(dim), cP)
    y[1] ~ MvGaussianMeanCovariance(s[1], cQ)
    
    for i in 2:n
        s[i] ~ MvGaussianMeanCovariance(cA * s[i - 1], cP)
        y[i] ~ MvGaussianMeanCovariance(s[i], cQ)
    end
    
    return s, y
end
```

The `~` expression can also return a reference to a newly created node in a corresponding factor graph for convenience in later usage:

```julia
@model function test_model()

    # In this example `ynode` refers to the corresponding 
    # `GaussianMeanVariance` node created in the factor graph
    ynode, y ~ GaussianMeanVariance(0.0, 1.0)
    
    return ynode, y
end
```

## Probabilistic inference in ReactiveMP.jl

`ReactiveMP.jl` uses the `Rocket.jl` package API for inference routines. `Rocket.jl` is a reactive programming extension for Julia that is higly inspired by `RxJS` and similar libraries from the `Rx` ecosystem. It consists of **observables**, **actors**, **subscriptions** and **operators**. For more infromation and rigorous examples see [Rocket.jl github page](https://github.com/biaslab/Rocket.jl).

### Observables
Observables are lazy push-based collections and they deliver their values over time.


```julia
# Timer that emits a new value every second and has an initial one second delay 
observable = timer(1000, 1000)
```




    TimerObservable(1000, 1000)



A subscription allows us to subscribe on future values of some observable, and actors specify what to do with these new values:


```julia
actor = (value) -> println(value)
subscription1 = subscribe!(observable, actor)
```




    TimerSubscription()




```julia
# We always need to unsubscribe from some observables
unsubscribe!(subscription1)
```


```julia
# We can modify our observables
modified = observable |> filter(d -> rem(d, 2) === 1) |> map(Int, d -> d ^ 2)
```




    ProxyObservable(Int64, MapProxy(Int64))




```julia
subscription2 = subscribe!(modified, (value) -> println(value))
```




    TimerSubscription()




```julia
unsubscribe!(subscription2)
```

The `ReactiveMP.jl` package returns posterior marginal distributions in our specified model in the form of an observable. It is possible to subscribe on its future updates, but for convenience `ReactiveMP.jl` only caches the last obtained values of all marginals in a model. To get a reference for the posterior marginal of some random variable in a model `ReactiveMP.jl` exports two functions: 
- `getmarginal(x)`: for a single random variable `x`
- `getmarginals(xs)`: for a dense sequence of random variables `sx`

Lets see how it works in practice. Here we create a simple coin toss model. We assume that observations are governed by the `Bernoulli` distribution with unknown bias parameter `θ`. To have a fully Bayesian treatment of this problem we endow `θ` with the `Beta` prior.


```julia
@model function coin_toss_model(n)

    # `datavar` creates data 'inputs' in our model
    # We will pass data later on to these inputs
    # In this example we create a sequence of inputs that accepts Float64
    y = datavar(Float64, n)
    
    # We endow θ parameter of our model with some prior
    θ ~ Beta(2.0, 7.0)
    
    # We assume that the outcome of each coin flip 
    # is modeled by a Bernoulli distribution
    for i in 1:n
        y[i] ~ Bernoulli(θ)
    end
    
    # We return references to our data inputs and θ parameter
    # We will use these references later on during the inference step
    return y, θ
end
```


```julia
_, (y, θ) = coin_toss_model(500);
```


```julia
# As soon as we have a new value for the marginal posterior over the `θ` variable
# we simply print the first two statistics of it
θ_subscription = subscribe!(getmarginal(θ), (marginal) -> println("New update: mean(θ) = ", mean(marginal), ", std(θ) = ", std(marginal)));
```

Next, lets define our dataset:


```julia
p = 0.75 # Bias of a coin

dataset = float.(rand(Bernoulli(p), 500));
```

To pass data to our model we use `update!` function


```julia
update!(y, dataset)
```

    New update: mean(θ) = 0.7426326129666012, std(θ) = 0.019358810889841



```julia
# It is necessary to always unsubscribe from running observables
unsubscribe!(θ_subscription)
```


```julia
# The ReactiveMP.jl inference backend is lazy and does not compute posterior marginals if no-one is listening for them
# At this moment we have already unsubscribed from the new posterior updates so this `update!` does nothing
update!(y, dataset)
```

`Rocket.jl` provides some useful built-in actors for obtaining posterior marginals especially with static datasets.


```julia
# the `keep` actor simply keeps all incoming updates in an internal storage, ordered
θvalues = keep(Marginal)
```




    KeepActor{Marginal}(Marginal[])




```julia
# `getmarginal` always emits last cached value as its first value
subscribe!(getmarginal(θ) |> take(1), θvalues);
```


```julia
getvalues(θvalues)
```




    1-element Vector{Marginal}:
     Marginal(Beta{Float64}(α=378.0, β=131.0))




```julia
subscribe!(getmarginal(θ) |> take(1), θvalues);
```


```julia
getvalues(θvalues)
```




    2-element Vector{Marginal}:
     Marginal(Beta{Float64}(α=378.0, β=131.0))
     Marginal(Beta{Float64}(α=378.0, β=131.0))




```julia
# the `buffer` actor keeps very last incoming update in an internal storage and can also store 
# an array of updates for a sequence of random variables
θbuffer = buffer(Marginal, 1)
```




    BufferActor{Marginal, Vector{Marginal}}(Marginal[#undef])




```julia
subscribe!(getmarginals([ θ ]) |> take(1), θbuffer);
```


```julia
getvalues(θbuffer)
```




    1-element Vector{Marginal}:
     Marginal(Beta{Float64}(α=378.0, β=131.0))




```julia
subscribe!(getmarginals([ θ ]) |> take(1), θbuffer);
```


```julia
getvalues(θbuffer)
```




    1-element Vector{Marginal}:
     Marginal(Beta{Float64}(α=378.0, β=131.0))



## Reactive Inference

ReactiveMP.jl naturally supports reactive streams of data and it is possible to run reactive inference with some external datasource.


```julia
@model function online_coin_toss_model()
    
    # We create datavars for the prior 
    # over `θ` variable
    θ_a = datavar(Float64)
    θ_b = datavar(Float64)
    
    θ ~ Beta(θ_a, θ_b)
    
    y = datavar(Float64)
    y ~ Bernoulli(θ)

    return θ_a, θ_b, θ, y
end

```


```julia
_, (θ_a, θ_b, θ, y) = online_coin_toss_model();
```


```julia
# In this example we subscribe on posterior marginal of θ variable and use it as a prior for our next observation
# We also print into stdout for convenience
θ_subscription = subscribe!(getmarginal(θ), (m) -> begin 
    m_a, m_b = params(m)
    update!(θ_a, m_a)
    update!(θ_b, m_b)
    println("New posterior for θ: mean = ", mean(m), ", std = ", std(m))
end);
```


```julia
# Initial priors
update!(θ_a, 10.0 * rand())
update!(θ_b, 10.0 * rand())
```


```julia
data_source = timer(500, 500) |> map(Float64, (_) -> float(rand(Bernoulli(0.75)))) |> tap((v) -> println("New observation: ", v));
```


```julia
data_subscription = subscribe!(data_source, (data) -> update!(y, data))
```




    TimerSubscription()




```julia
# It is important to unsubscribe from running observables to release computer resources
unsubscribe!(data_subscription)
unsubscribe!(θ_subscription)
```

That was an example of exact Bayesian inference with Sum-Product (or Belief Propagation) algorithm. However, `ReactiveMP.jl` is not limited to only the sum-product algorithm but it also supports variational message passing with [Constrained Bethe Free Energy Minimisation](https://www.mdpi.com/1099-4300/23/7/807).

## Variational inference

On a very high-level, ReactiveMP.jl is aimed to solve the Constrained Bethe Free Energy minimisation problem. For this task we approximate our exact posterior marginal distribution by some family of distributions $q \in \mathcal{Q}$. Often this involves assuming some factorization over $q$. For this purpose the `@model` macro supports optional `where { ... }` clauses for every `~` expression in a model specification.


```julia
@model function test_model6_with_manual_constraints(n)
    τ ~ GammaShapeRate(1.0, 1.0) 
    μ ~ NormalMeanVariance(0.0, 100.0)
    
    y = datavar(Float64, n)
    
    for i in 1:n
        # Here we assume a mean-field assumption on our 
        # variational family of distributions locally for the current node
        y[i] ~ NormalMeanPrecision(μ, τ) where { q = q(y[i])q(μ)q(τ) }
    end
    
    return μ, τ, y
end
```

In this example we specified an extra constraints for $q_a$ for Bethe factorisation:

```math
q(s) = \prod_{a \in \mathcal{V}} q_a(s_a) \prod_{i \in \mathcal{E}} q_i^{-1}(s_i)
```

There are several options to specify the mean-field factorisation constraint. 

```julia
y[i] ~ NormalMeanPrecision(μ, τ) where { q = q(y[i])q(μ)q(τ) } # With names from model specification
y[i] ~ NormalMeanPrecision(μ, τ) where { q = q(out)q(mean)q(precision) } # With names from node specification
y[i] ~ NormalMeanPrecision(μ, τ) where { q = MeanField() } # With alias name
```

It is also possible to use local structured factorisation:

```julia
y[i] ~ NormalMeanPrecision(μ, τ) where { q = q(y[i], μ)q(τ) } # With names from model specification
y[i] ~ NormalMeanPrecision(μ, τ) where { q = q(out, mean)q(precision) } # With names from node specification
```

As an option the `@model` macro accepts optional arguments for model specification, one of which is `default_factorisation` that accepts `MeanField()` as its argument for better convenience

```julia
@model [ default_factorisation = MeanField() ] function test_model(...)
    ...
end
```
This will autatically impose a mean field factorization constraint over all marginal distributions in our model.

### GraphPPL.jl constraints macro

`GraphPPL.jl` package exports `@constraints` macro to simplify factorisation and form constraints specification. Read more about `@constraints` macro in the corresponding documentation section, here we show a simple example of the same factorisation constraints specification, but with `@constraints` macro:


```julia
constraints6 = @constraints begin
     q(μ, τ) = q(μ)q(τ) # Mean-Field over `μ` and `τ`
end
```




    Constraints:
    	marginals form:
    	messages form:
    	factorisation:
    		q(μ, τ) = q(μ)q(τ)




**Note**: `where` blocks have higher priority over constraints specification


```julia
@model function test_model6(n)
    τ ~ GammaShapeRate(1.0, 1.0) 
    μ ~ NormalMeanVariance(0.0, 100.0)
    
    y = datavar(Float64, n)
    
    for i in 1:n
        # Here we assume a mean-field assumption on our 
        # variational family of distributions locally for the current node
        y[i] ~ NormalMeanPrecision(μ, τ)
    end
    
    return μ, τ, y
end
```

### Inference

To run inference in this model we again need to create a synthetic dataset:


```julia
dataset = rand(Normal(-3.0, inv(sqrt(5.0))), 1000);
```

#### `inference` function

In order to simplify model and inference testing, `ReactiveMP.jl` exports pre-written inference function, that is aimed for simple use cases with static datasets:


```julia
?inference
```

```
inference(
    # `model`: specifies a model generator, with the help of the `Model` function
    model::ModelGenerator; 
    # NamedTuple or Dict with data, required
    data,
    # NamedTuple with initial marginals, optional, defaults to empty
    initmarginals = nothing,
    # NamedTuple with initial messages, optional, defaults to empty
    initmessages = nothing,  # optional
    # Constraints specification object
    constraints = nothing,
    # Meta specification object
    meta  = nothing,
    # Model creation options
    options = (;),
    # Return structure info, optional, defaults to return everything at each iteration
    returnvars = nothing, 
    # Number of iterations, defaults to 1, we do not distinguish between VMP or Loopy belief or EP iterations
    iterations = 1,
    # Do we compute FE, optional, defaults to false
    free_energy = false,
    # Show progress module, optional, defaults to false
    showprogress = false,
)
```

This function provides generic (but somewhat limited) way to run inference in ReactiveMP.jl. 





```julia
result = inference(
    model         = Model(test_model6, length(dataset)),
    data          = (y = dataset, ),
    constraints   = constraints6, 
    initmarginals = (μ = vague(NormalMeanPrecision), τ = vague(GammaShapeRate)),
    returnvars    = (μ = KeepLast(), τ = KeepLast()),
    iterations    = 10,
    free_energy   = true,
    showprogress  = true
)
```

    Inference results:
    -----------------------------------------
    Free Energy: Real[14763.3, 3276.07, 645.285, 601.821, 601.821, 601.821, 601.821, 601.821, 601.821, 601.821]
    -----------------------------------------
    μ = Marginal(NormalWeightedMeanPrecision{Float64}(xi=-15682.23170282978, w=5204.7373...
    τ = Marginal(GammaShapeRate{Float64}(a=501.0, b=96.25864505252237))





```julia
println("μ: mean = ", mean(result.posteriors[:μ]), ", std = ", std(result.posteriors[:μ]))
```

    μ: mean = -3.0130688119259657, std = 0.013861192405657346



```julia
println("τ: mean = ", mean(result.posteriors[:τ]), ", std = ", std(result.posteriors[:τ]))
```

    τ: mean = 5.204727323209624, std = 0.23253006806179705


#### Manual inference

For advanced use cases it is advised to write inference functions manually as it provides more flexibility, here is an example of manual inference specification:


```julia
model, (μ, τ, y) = test_model6(constraints6, length(dataset));
```

For variational inference we also usually need to set initial marginals for our inference procedure. For that purpose `ReactiveMP.jl` export the `setmarginal!` function:


```julia
setmarginal!(μ, vague(NormalMeanPrecision))
setmarginal!(τ, vague(GammaShapeRate))
```


```julia
μ_values = keep(Marginal)
τ_values = keep(Marginal)

μ_subscription = subscribe!(getmarginal(μ), μ_values)
τ_subscription = subscribe!(getmarginal(τ), τ_values)

for i in 1:10
    update!(y, dataset)
end
```


```julia
getvalues(μ_values)
```




    10-element Vector{Marginal}:
     Marginal(NormalWeightedMeanPrecision{Float64}(xi=-3.019100750200221e-9, w=0.010000001002000566))
     Marginal(NormalWeightedMeanPrecision{Float64}(xi=-27.629495773280823, w=9.179867803429282))
     Marginal(NormalWeightedMeanPrecision{Float64}(xi=-10021.247182136565, w=3325.930698652331))
     Marginal(NormalWeightedMeanPrecision{Float64}(xi=-15673.395541852733, w=5201.80471710224))
     Marginal(NormalWeightedMeanPrecision{Float64}(xi=-15682.22287937003, w=5204.734394819007))
     Marginal(NormalWeightedMeanPrecision{Float64}(xi=-15682.231694023943, w=5204.737320287038))
     Marginal(NormalWeightedMeanPrecision{Float64}(xi=-15682.231702820985, w=5204.737323206587))
     Marginal(NormalWeightedMeanPrecision{Float64}(xi=-15682.231702829797, w=5204.737323209579))
     Marginal(NormalWeightedMeanPrecision{Float64}(xi=-15682.231702829771, w=5204.737323209567))
     Marginal(NormalWeightedMeanPrecision{Float64}(xi=-15682.23170282978, w=5204.737323209567))




```julia
getvalues(τ_values)
```




    10-element Vector{Marginal}:
     Marginal(GammaShapeRate{Float64}(a=501.0, b=5.0000000000463575e14))
     Marginal(GammaShapeRate{Float64}(a=501.0, b=54635.465934702996))
     Marginal(GammaShapeRate{Float64}(a=501.0, b=150.63498062446408))
     Marginal(GammaShapeRate{Float64}(a=501.0, b=96.31291260934161))
     Marginal(GammaShapeRate{Float64}(a=501.0, b=96.25869921157143))
     Marginal(GammaShapeRate{Float64}(a=501.0, b=96.25864510657314))
     Marginal(GammaShapeRate{Float64}(a=501.0, b=96.25864505257626))
     Marginal(GammaShapeRate{Float64}(a=501.0, b=96.25864505252225))
     Marginal(GammaShapeRate{Float64}(a=501.0, b=96.25864505252243))
     Marginal(GammaShapeRate{Float64}(a=501.0, b=96.25864505252237))




```julia
println("μ: mean = ", mean(last(μ_values)), ", std = ", std(last(μ_values)))
```

    μ: mean = -3.0130688119259657, std = 0.013861192405657346



```julia
println("τ: mean = ", mean(last(τ_values)), ", std = ", std(last(τ_values)))
```

    τ: mean = 5.204727323209624, std = 0.23253006806179705


### Form constraints

In order to support form constraints, the `randomvar()` function also supports a `where { ... }` clause with some optional arguments. One of these arguments is `form_constraint` that allows us to specify a form constraint to the random variables in our model. Another one is `prod_constraint` that allows to specify an additional constraints during computation of product of two colliding messages. For example we can perform the EM algorithm if we assign a point mass contraint on some variables in our model.


```julia
@model function test_model7_with_manual_constraints(n)
    τ ~ GammaShapeRate(1.0, 1.0) 
    
    # In case of form constraints `randomvar()` call is necessary
    μ = randomvar() where { marginal_form_constraint = PointMassFormConstraint() }
    μ ~ NormalMeanVariance(0.0, 100.0)
    
    y = datavar(Float64, n)
    
    for i in 1:n
        y[i] ~ NormalMeanPrecision(μ, τ) where { q = q(y[i])q(μ)q(τ) }
    end
    
    return μ, τ, y
end
```

As in the previous example we can use `@constraints` macro to achieve the same goal with a nicer syntax:


```julia
constraints7 = @constraints begin 
    q(μ) :: PointMass
    
    q(μ, τ) = q(μ)q(τ) # Mean-Field over `μ` and `τ`
end
```




    Constraints:
    	marginals form:
    		q(μ) :: PointMassFormConstraint() [ prod_constraint = ProdGeneric(ProdAnalytical())]
    	messages form:
    	factorisation:
    		q(μ, τ) = q(μ)q(τ)




In this example we specified an extra constraints for $q_i$ for Bethe factorisation:

```math
q(s) = \prod_{a \in \mathcal{V}} q_a(s_a) \prod_{i \in \mathcal{E}} q_i^{-1}(s_i)
```


```julia
@model function test_model7(n)
    τ ~ GammaShapeRate(1.0, 1.0) 
    
    # In case of form constraints `randomvar()` call is necessary
    μ = randomvar()
    μ ~ NormalMeanVariance(0.0, 100.0)
    
    y = datavar(Float64, n)
    
    for i in 1:n
        y[i] ~ NormalMeanPrecision(μ, τ)
    end
    
    return μ, τ, y
end
```


```julia
model, (μ, τ, y) = test_model7(constraints7, length(dataset));
```


```julia
setmarginal!(μ, vague(NormalMeanPrecision))
setmarginal!(τ, PointMass(1.0))

μ_values = keep(Marginal)
τ_values = keep(Marginal)

μ_subscription = subscribe!(getmarginal(μ), μ_values)
τ_subscription = subscribe!(getmarginal(τ), τ_values)

for i in 1:10
    update!(y, dataset)
end
```


```julia
getvalues(μ_values) |> last
```




    Marginal(PointMass{Float64}(-3.0130688177034712))




```julia
getvalues(τ_values) |> last 
```




    Marginal(GammaShapeRate{Float64}(a=501.0, b=96.1625787250356))



By default `ReactiveMP.jl` tries to compute an analytical product of two colliding messages and throws an error if no analytical solution is known. However, it is possible to fall back to a generic product that does not require an analytical solution to be known. In this case the inference backend will simply propagate the product of two message in a form of a tuple. It is not possible to use such a tuple-product during an inference and in this case it is mandatory to use some form constraint to approximate this product.

```julia
μ = randomvar() where { 
    prod_constraint = ProdGeneric(),
    form_constraint = SampleListFormConstraint() 
}
```

Sometimes it is useful to preserve a specific parametrisation of the resulting product later on in an inference procedure. `ReactiveMP.jl` exports a special `prod_constraint` called `ProdPreserveType` especially for that purpose:

```julia
μ = randomvar() where { prod_constraint = ProdPreserveType(NormalWeightedMeanPrecision) }
```

**Note**: `@constraints` macro specifies required `prod_constraint` automatically.

### Free Energy

During variational inference `ReactiveMP.jl` optimises a special functional called the Bethe Free Energy functional. It is possible to obtain its values for all VMP iterations with the `score` function.


```julia
model, (μ, τ, y) = test_model6(constraints6, length(dataset));
```


```julia
bfe_observable = score(BetheFreeEnergy(), model)
```




    ProxyObservable(Real, MapProxy(Tuple{ReactiveMP.InfCountingReal, ReactiveMP.InfCountingReal}))




```julia
bfe_subscription = subscribe!(bfe_observable, (fe) -> println("Current BFE value: ", fe));
```


```julia
# Reset the model with vague marginals
setmarginal!(μ, vague(NormalMeanPrecision))
setmarginal!(τ, vague(GammaShapeRate))

for i in 1:10
    update!(y, dataset)
end
```

    Current BFE value: 614.3116250537296
    Current BFE value: 601.8211489410164
    Current BFE value: 601.8211486920968
    Current BFE value: 601.8211486920954
    Current BFE value: 601.8211486920986
    Current BFE value: 601.8211486920968
    Current BFE value: 601.8211486920968
    Current BFE value: 601.8211486920964
    Current BFE value: 601.8211486920964
    Current BFE value: 601.8211486920968



```julia
# It always necessary to unsubscribe and release computer resources
unsubscribe!([ μ_subscription, τ_subscription, bfe_subscription ])
```

### Meta data specification

During model specification some functional dependencies may accept an optional `meta` object in the `where { ... }` clause. The purpose of the `meta` object is to adjust, modify or supply some extra information to the inference backend during the computations of the messages. The `meta` object for example may contain an approximation method that needs to be used during various approximations or it may specify the tradeoff between accuracy and performance:

```julia
# In this example the `meta` object for the autoregressive `AR` node specifies the variate type of 
# the autoregressive process and its order. In addition it specifies that the message computation rules should
# respect accuracy over speed with the `ARsafe()` strategy. In contrast, `ARunsafe()` strategy tries to speedup computations
# by cost of possible numerical instabilities during an inference procedure
s[i] ~ AR(s[i - 1], θ, γ) where { q = q(s[i - 1], s[i])q(θ)q(γ), meta = ARMeta(Multivariate, order, ARsafe()) }
...
s[i] ~ AR(s[i - 1], θ, γ) where { q = q(s[i - 1], s[i])q(θ)q(γ), meta = ARMeta(Univariate, order, ARunsafe()) }
```

Another example with `GaussianControlledVariance`, or simply `GCV` [see Hierarchical Gaussian Filter], node:

```julia
# In this example we specify structured factorisation and flag meta with `GaussHermiteCubature` 
# method with `21` sigma points for approximation of non-lineariety between hierarchy layers
xt ~ GCV(xt_min, zt, real_k, real_w) where { q = q(xt, xt_min)q(zt)q(κ)q(ω), meta = GCVMetadata(GaussHermiteCubature(21)) }
```

The Meta object is useful to pass any extra information to a node that is not a random variable or constant model variable. It may include extra approximation methods, differentiation methods, optional non-linear functions, extra inference parameters etc.

### GraphPPL.jl `@meta` macro

Users can use `@meta` macro from the `GraphPPL.jl` package to achieve the same goal. Read more about `@meta` macro in the corresponding documentation section. Here is a simple example of the same meta specification:


```julia
@meta begin 
     AR(s, θ, γ) -> ARMeta(Multivariate, 5, ARsafe())
end
```




    Meta specification:
    	AR(s, θ, γ) = ARMeta{Multivariate, ARsafe}(5, ARsafe())



## Creating custom nodes and message computation rules

### Custom nodes

To create a custom functional form and to make it available during model specification `ReactiveMP.jl` exports the `@node` macro:

```julia
# `@node` macro accepts a name of the functional form, its type, either `Stochastic` or `Deterministic` and an array of interfaces:
@node NormalMeanVariance Stochastic [ out, μ, v ]

# Interfaces may have aliases for their names that might be convenient for factorisation constraints specification
@node NormalMeanVariance Stochastic [ out, (μ, aliases = [ mean ]), (v, aliases = [ var ]) ]

# `NormalMeanVariance` structure declaration must exist, otherwise `@node` macro will throw an error
struct NormalMeanVariance end 

@node NormalMeanVariance Stochastic [ out, μ, v ]

# It is also possible to use function objects as a node functional form
function dot end

# Syntax for functions is a bit differet, as it is necesssary to use `typeof(...)` function for them 
# out = dot(x, a)
@node typeof(dot) Deterministic [ out, x, a ]
```

**Note**: Deterministic nodes do not support factorisation constraints with the `where { q = ... }` clause.

After that it is possible to use the newly created node during model specification:

```julia
@model function test_model()
    ...
    y ~ dot(x, a)
    ...
end
```

### Custom messages computation rules

`ReactiveMP.jl` exports the `@rule` macro to create custom message computation rules. For example let us create a simple `+` node to be available for usage in the model specification usage. We refer to *A Factor Graph Approach to Signal Modelling , System Identification and Filtering* [ Sascha Korl, 2005, page 32 ] for a rigorous explanation of the `+` node in factor graphs. According to Korl, assuming that inputs are Gaussian Sum-Product message computation rule for `+` node is the following:

```math
\mu_z = \mu_x + \mu_y \\
V_z = V_x + V_y
```

To specify this in `ReactiveMP.jl` we use the `@node` and `@rule` macros:
 
```julia
@node typeof(+) Deterministic  [ z, x, y ]

@rule typeof(+)(:z, Marginalisation) (m_x::UnivariateNormalDistributionsFamily, m_y::UnivariateNormalDistributionsFamily) = begin
    x_mean, x_var = mean_var(m_x)
    y_mean, y_var = mean_var(m_y)
    return NormalMeanVariance(x_mean + y_mean, x_var + y_var)
end
```

In this example, for the `@rule` macro, we specify a type of our functional form: `typeof(+)`. Next, we specify an edge we are going to compute an outbound message for. `Marginalisation` indicates that the corresponding message respects the marginalisation constraint for posterior over corresponding edge:

```math
q(z) = \int q(z, x, y) \mathrm{d}x\mathrm{d}y
```

If we look on difference between sum-product rules and variational rules with mean-field assumption we notice that they require different local information to compute an outgoing message:

<div style="width:100%">
```math
\mu(z) = \int f(x, y, z)\mu(x)\mu(y)\mathrm{d}x\mathrm{d}y
```

```math
\nu(z) = \exp{ \int \log f(x, y, z)q(x)q(y)\mathrm{d}x\mathrm{d}y }
```
</div>

The `@rule` macro supports both cases with special prefixes during rule specification:
- `m_` prefix corresponds to the incoming message on a specific edge
- `q_` prefix corresponds to the posterior marginal of a specific edge

Example of a Sum-Product rule with `m_` messages used:

```julia
@rule NormalMeanPrecision(:μ, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_τ::PointMass) = begin 
    m_out_mean, m_out_cov = mean_cov(m_out)
    return NormalMeanPrecision(m_out_mean, inv(m_out_cov + inv(mean(m_τ))))
end
```

Example of a Variational rule with Mean-Field assumption with `q_` posteriors used:

```julia
@rule NormalMeanPrecision(:μ, Marginalisation) (q_out::Any, q_τ::Any) = begin 
    return NormalMeanPrecision(mean(q_out), mean(q_τ))
end
```

`ReactiveMP.jl` also supports structured rules. It is possible to obtain joint marginal over a set of edges:

```julia
@rule NormalMeanPrecision(:τ, Marginalisation) (q_out_μ::Any, ) = begin
    m, V = mean_cov(q_out_μ)
    θ = 2 / (V[1,1] - V[1,2] - V[2,1] + V[2,2] + abs2(m[1] - m[2]))
    α = convert(typeof(θ), 1.5)
    return Gamma(α, θ)
end
```

**NOTE**: In the `@rule` specification the messages or marginals arguments **must** be in order with interfaces specification from `@node` macro:

```julia
# Inference backend expects arguments in `@rule` macro to be in the same order
@node NormalMeanPrecision Stochastic [ out, μ, τ ]
```

Any rule always has access to the meta information with hidden the `meta::Any` variable:

```julia
@rule MyCustomNode(:out, Marginalisation) (m_in1::Any, m_in2::Any) = begin 
    ...
    println(meta)
    ...
end
```

It is also possible to dispatch on a specific type of a meta object:

```julia
@rule MyCustomNode(:out, Marginalisation) (m_in1::Any, m_in2::Any, meta::LaplaceApproximation) = begin 
    ...
end
```

or

```julia
@rule MyCustomNode(:out, Marginalisation) (m_in1::Any, m_in2::Any, meta::GaussHermiteCubature) = begin 
    ...
end
```

### Customizing messages computational pipeline

In certain situations it might be convenient to customize the default message computational pipeline. `GrahpPPL.jl` supports the `pipeline` keyword in the `where { ... }` clause to add some extra steps after a message has been computed. A use case might be an extra approximation method to preserve conjugacy in the model, debugging or simple printing.

```julia
# Logs all outbound messages
y[i] ~ NormalMeanPrecision(x[i], 1.0) where { pipeline = LoggerPipelineStage() }
# Initialise messages to be vague
y[i] ~ NormalMeanPrecision(x[i], 1.0) where { pipeline = InitVaguePipelineStage() }
# In principle, it is possible to approximate outbound messages with Laplace Approximation
y[i] ~ NormalMeanPrecision(x[i], 1.0) where { pipeline = LaplaceApproximation() }
```

Let us return to the coin toss model, but this time we want to print flowing messages:


```julia
@model [ default_factorisation = FullFactorisation() ] function coin_toss_model_log(n)

    y = datavar(Float64, n)

    θ ~ Beta(2.0, 7.0) where { pipeline = LoggerPipelineStage("θ") }

    for i in 1:n
        y[i] ~ Bernoulli(θ)  where { pipeline = LoggerPipelineStage("y[$i]") }
    end
    
    return y, θ
end
```


```julia
_, (y, θ) = coin_toss_model_log(5);
```


```julia
θ_subscription = subscribe!(getmarginal(θ), (value) -> println("New posterior marginal for θ: ", value));
```

    [θ][Beta][out]: Message(Beta{Float64}(α=2.0, β=7.0))



```julia
coinflips = float.(rand(Bernoulli(0.5), 5));
```


```julia
update!(y, coinflips)
```

    [y[1]][Bernoulli][p]: Message(Beta{Float64}(α=2.0, β=1.0))
    [y[2]][Bernoulli][p]: Message(Beta{Float64}(α=1.0, β=2.0))
    [y[3]][Bernoulli][p]: Message(Beta{Float64}(α=1.0, β=2.0))
    [y[4]][Bernoulli][p]: Message(Beta{Float64}(α=2.0, β=1.0))
    [y[5]][Bernoulli][p]: Message(Beta{Float64}(α=2.0, β=1.0))
    New posterior marginal for θ: Marginal(Beta{Float64}(α=5.0, β=9.0))



```julia
unsubscribe!(θ_subscription)
```


```julia
# Inference is lazy and does not send messages if no one is listening for them
update!(y, coinflips)
```


```julia

```
