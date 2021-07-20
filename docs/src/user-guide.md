# User guide

This user guide outlines the usage of `ReactiveMP` for solving inference problems. The content is divided in several parts:

- [Specifying a model](@ref user-guide-model-specification)
- [Specifying an inference procedure](@ref user-guide-inference-specification)
- [Inference execution](@ref user-guide-inference-execution)

## [User guide: Model Specification](@id user-guide-model-specification)

Probabilistic models incorporate elements of randomness to describe an event or phenomenon by using random variables and probability theory. A probabilistic model can be represented visually by using probabilistic graphical models (PGMs). A factor graph is a type of PGM that is well suited to cast inference tasks in terms of graphical manipulations.

`GraphPPL.jl` is a Julia package presenting a model specification language for probabilistic models.

## Model specification

The `ReactiveMP` uses `GraphPPL` library to simplify model specification. It is not necessary but highly recommended to use `ReactiveMP` in a combination with `GraphPPL` model specification library.
The `GraphPPL` library exports a single `@model` macro for model specification. The `@model` macro accepts two arguments: model options (optionally) and the model specification itself in a form of regular Julia function. 

For example: 

```julia
# `@model` macro accepts an array of named options as a first argument and
# a regular Julia function body as its second argument
@model [ option1 = ..., option2 = ... ] function model_name(model_arguments...)
    # model specification goes here
    return ...
end
```

Model options are optional and may be omitted:

```julia
@model function model_name(model_arguments...)
    # model specification here
    return ...
end
```

that is equivalent to 

```julia
# Empty options if ommited
@model [] function model_name(model_arguments...)
    # model specification here
    return ...
end
```

The `@model` macro returns a regular Julia function (in this example `model_name(model_arguments...)`) that has the same signature and can be executed as usual. It returns a reference to a model object itself and a tuple of a user specified return variables, e.g:

```julia
@model function my_model(model_arguments...)
    # model specification here
    # ...
    return x, y
end
```

```julia
model, (x, y) = my_model(model_arguments...)
```

It is also important to note that any model should return something, such as variables or nodes. If a model doesn't return anything then an error will be raised during runtime. 
`model` object might be useful to inspect model's factor graph and/or factor nodes and variables. It is also used in Bethe Free Energy score computation. If not needed it can be ommited with `_` placeholder, eg:

```julia
_, (x, y) = my_model(model_arguments...)
```

## A full example before diving in

Before presenting the details of the model specification syntax, we show an example of a simple probabilistic model.
Here we create a linear gaussian state space model with latent random variables `x` and noisy observations `y`:

```julia
@model [ options... ] function state_space_model(n_observations, noise_variance)

    x = randomvar(n_observations)
    y = datavar(Float64, n_observations)

    x[1] ~ NormalMeanVariance(0.0, 100.0)

    for i in 2:n_observations
       x[i] ~ x[i - 1] + 1.0
       y[i] ~ NormalMeanVariance(x[i], noise_variance)
    end

    return x, y
end
```
    
## Graph variables creation

### Constants

Any runtime constant passed to a model as a model argument will be automatically converted to a fixed constant in the graph model. This convertion happens every time when model specification identifies a constant. Sometimes it might be useful to create constants by hand (e.g. to avoid copying large matrices across the model and to avoid extensive memory allocations).

You can create a constant within a model specification macro with `constvar()` function. For example:

```julia
c = constvar(1.0)

for i in 2:n
    x[i] ~ x[i - 1] + c # Reuse the same reference to a constant 1.0
end
```

Additionally you can specify an extra `::ConstVariable` type for some of the model arguments. In this case macro automatically converts them to a single constant using `constvar()` function. E.g.:

```julia
@model function model_name(nsamples::Int, c::ConstVariable)
    # ...
    # no need to call for a constvar() here
    for i in 2:n
        x[i] ~ x[i - 1] + c # Reuse the same reference to a constant `c`
    end
    # ...
    return ...
end
```

!!! note
    `::ConstVariable` does not restrict an input type of an argument and does not interfere with multiple dispatch. In this example `c` can have any type, e.g. `Int`.

### Data variables

It is important to have a mechanism to pass data values to the model. You can create data inputs with `datavar()` function. As a first argument it accepts a type specification and optional dimensionality (as additional arguments or as a tuple).

Examples: 

```julia
y = datavar(Float64) # Creates a single data input with `y` as identificator
y = datavar(Float64, n) # Returns a vector of  `y_i` data input objects with length `n`
y = datavar(Float64, n, m) # Returns a matrix of `y_i_j` data input objects with size `(n, m)`
y = datavar(Float64, (n, m)) # It is also possible to use a tuple for dimensionality, it is an equivalent of the previous line
```

### Random variables

There are several ways to create random variables. The first one is an explicit call to `randomvar()` function. By default it doesn't accept any argument, creates a single random variable in the model and returns it. It is also possible to pass dimensionality arguments to `randomvar()` function in the same way as for the `datavar()` function.

Examples: 

```julia
x = randomvar() # Returns a single random variable which can be used later in the model
x = randomvar(n) # Returns an vector of random variables with length `n`
x = randomvar(n, m) # Returns a matrix of random variables with size `(n, m)`
x = randomvar((n, m)) # It is also possible to use a tuple for dimensionality, it is an equivalent of the previous line
```

The second way to create a random variable is to use the `~` operator. If the random variable hasn't been created yet, `~` operator will be creat it automatically during the creation of the node. Read more about the `~` operator in the next section.

## Node creation

Factor nodes (or local functions) are used to define a relationship between random variables and/or constants and data inputs. In most of the cases a factor node defines a probability distribution over selected random variables. 

We model a random variable by a probability distribution using the `~` operator. For example, to create a random variable `y` which is modeled by a Normal distribution, where its mean and variance are controlled by the random variables `m` and `v` respectively, we define

```julia
m = randomvar()
v = randomvar()
y ~ NormalMeanVariance(m, v) # Creates a `y` random variable automatically
```

It is also possible to use a deterministic relationships between random variables:

```julia
a = randomvar()
b = randomvar()
c ~ a + b # Here with the help of `~` operator we explictly say that `c` is a random variable too
```

!!! note
    The `GraphPPL.jl` package uses the `~` operator for modelling both stochastic and deterministic relationships between random variables.


The `@model` macro automatically resolves any inner function calls into anonymous extra nodes. It is also worth to note that inference backend will try to optimize inner deterministic function calls in the case where all arguments are constants or data inputs. For example:

```julia
noise ~ NormalMeanVariance(mean, inv(precision)) # Will create a non-linear node `inv` in case if `precision` is a random variable. Won't create an additional non-linear node in case if `precision` is a constant or data input.
```

It is possible to use any functional expression within the `~` operator arguments list. The only one exception is the `ref` expression (e.g `x[i]` or `x[i, j]`). In principle `x[i]` expression is equivalent to `getindex(x, i)` and therefore might be treated as a factor node with `getindex` as local function, however all `ref` expressions within the `~` operator arguments list are left untouched during model parsing. This means that the model parser will not create unnecessary nodes when only simple indexing is involved.

```julia
y ~ NormalMeanVariance(x[i - 1], variance) # While in principle `x[i - 1]` is equivalent to (`getindex(x, -(i, 1))`) model parser will leave it untouched and won't create any anonymous nodes for this expression.

y ~ NormalMeanVariance(A * x[i - 1], variance) # This example will create a `*` anonymous node (in case if x[i - 1] is a random variable) and leave `x[i - 1]` untouched.
```

It is also possible to return a node reference from the `~` operator with the following syntax:

```julia
node, y ~ NormalMeanVariance(mean, var)
```

Having a node reference can be useful in case the user wants to return it from a model and to use it later on to specify initial joint marginal distributions.

### Node creation options

To pass optional arguments to the node creation constructor the user can use the `where { options...  }` specification syntax.

Example:

```julia
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(y_mean)q(y_var)q(y) } # mean-field factorisation over q
```

A list of all available options is presented below:

#### Factorisation constraint option

Users can specify a factorisation constraint over the approximate posterior `q` for variational inference.
The general syntax for factorisation constraints over `q` is the following:

```julia
variable ~ Node(node_arguments...) where { q = RecognitionFactorisationConstraint }
```

where `RecognitionFactorisationConstraint` can be one the following:

1. `MeanField()`

Automatically specifies a mean-field factorisation

Example:

```julia
y ~ NormalMeanVariance(y_mean, y_var) where { q = MeanField() }
```

2. `FullFactorisation()`

Automatically specifies a full factorisation (this is the default)

Example:

```julia
y ~ NormalMeanVariance(y_mean, y_var) where { q = FullFactorisation() }
```

3. `q(μ)q(v)q(out)` or `q(μ) * q(v) * q(out)`

A user can specify any factorisation he wants as the multiplication of `q(interface_names...)` factors. As interface names the user can use the interface names of an actual node (read node's documentation), its aliases (if available) or actual random variable names present in the `~` operator expression.

Examples: 

```julia
# Using interface names of a `NormalMeanVariance` node for factorisation constraint. 
# Call `?NormalMeanVariance` to know more about interface names for some node
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(μ)q(v)q(out) }
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(μ, v)q(out) }

# Using interface names aliases of a `NormalMeanVariance` node for factorisation constraint. 
# Call `?NormalMeanVariance` to know more about interface names aliases for some node
# In general aliases correspond to the function names for distribution parameters
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(mean)q(var)q(out) }
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(mean, var)q(out) }

# Using random variables names from `~` operator expression
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(y_mean)q(y_var)q(y) }
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(y_mean, y_var)q(y) }

# All methods can be combined easily
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(μ)q(y_var)q(out) }
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(y_mean, v)q(y) }
```

#### Metadata option

Is is possible to pass any extra metadata to a factor node with the `meta` option (if node supports it, read node's documentation). Metadata can be later accessed in message computation rules:

```julia
z ~ f(x, y) where { meta = ... }
```

#### Pipeline option

To assign a factor node's local pipeline we use a `pipeline` option:

```julia
y ~ NormalMeanVariance(m, v) where { pipeline = LoggerPipelineStage() } # Logs all outbound messages with `LoggerPipelineStage`
```

## [User guide: Inference specification](@id user-guide-inference-specification)

## [User guide: Inference execution](@id user-guide-inference-execution)