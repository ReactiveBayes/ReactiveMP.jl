# [Constraints Specification](@id user-guide-constraints-specification)

`GraphPPL.jl` exports `@constraints` macro for the extra constraints specification that can be used during the inference step in `ReactiveMP.jl` package.

### General syntax 

`@constraints` macro accepts both regular julia functions and just simple blocks. In the first case it returns a function that return constraints and in the second case it returns constraints directly.

```julia
myconstraints = @constraints begin 
    q(x) :: PointMass
    q(x, y) = q(x)q(y)
end
```

or 

```julia
@constraints function make_constraints(flag)
    q(x) :: PointMass
    if flag
        q(x, y) = q(x)q(y)
    end
end

myconstraints = make_constraints(true)
```

### Marginal and messages form constraints

To specify marginal or messages form constraints `@constraints` macro uses `::` operator (in the similar way as Julia uses it for type specification)

The following constraint

```julia
@constraints begin 
    q(x) :: PointMass
end
```

indicates that the resulting marginal of the variable (or array of variables) named `x` must be approximated with a `PointMass` object. To set messages form constraint `@constraints` macro uses `μ(...)` instead of `q(...)`:

```julia
@constraints begin 
    q(x) :: PointMass
    μ(x) :: SampleList 
    # it is possible to assign different form constraints on the same variable 
    # both for the marginal and for the messages 
end
```

`@constraints` macro understands "stacked" form constraints. For example the following form constraint

```julia
@constraints begin 
    q(x) :: SampleList(1000, LeftProposal()) :: PointMass
end
```

indicates that the resulting posterior first maybe approximated with a `SampleList` and in addition the result of this approximation should be approximated as a `PointMass`. 


### Factorisation constraints on posterior distribution `q()`

`@model` macro specifies generative model `p(s, y)` where `s` is a set of random variables and `y` is a set of obseervations. In a nutshell the goal of probabilistic programming is to find `p(s|y)`. `p(s|y)` during the inference procedure can be approximated with another `q(s)` using e.g. KL divergency. By default there are no extra factorisation constraints on `q(s)` and the result is `q(s) = p(s|y)`. However, inference may be not tractable for every model without extra factorisation constraints. To circumvent this, `GraphPPL.jl` and `ReactiveMP.jl` accepts optional factorisation constraints specification syntax:

For example:

```julia
@constraints begin 
    q(x, y) = q(x)q(y)
end
```

specifies a so-called mean-field assumption on variables `x` and `y` in the model. Futhermore, if `x` is an array of variables in our model we may induce extra mean-field assumption on `x` in the following way.

```julia
@constraints begin 
    q(x, y) = q(x)q(y)
    q(x) = q(x[begin])..q(x[end])
end
```

These constraints specifies a mean-field assumption between variables `x` and `y` (either single variable or collection of variables) and additionally specifies mean-field assumption on variables `x_i`.

!!! note 
    `@constraints` macro does not support matrix-based collections of variables. E.g. it is not possible to write `q(x[begin, begin])..q(x[end, end])`

It is possible to write more complex factorisation constraints, for example:

```julia
@constraints begin 
    q(x, y) = q(x[begin], y[begin])..q(x[end], y[end])
end
```

Specifies a mean-field assumption between collection of variables named `x` and `y` only for variables with different indices. Another example is

```julia
@constraints function make_constraints(k)
    q(x) = q(x[begin:k])q(x[k+1:end])
end
```

In this example we specify a mean-field assumption between a set of variables `x[begin:k]` and `x[k+1:end]`. 

To create a model with extra constraints user may use optional `constraints` keyword argument for the model function:

```julia
@model function my_model(arguments...)
   ...
end

constraints = @constraints begin 
    ...
end

model, (x, y) = model_name(arguments..., constraints = constraints)
```