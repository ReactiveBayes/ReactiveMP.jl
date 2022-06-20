# [Custom Functional Form Specification](@id custom-functional-form)

In a nutshell, functional form constraints defines a function that approximates the product of colliding messages and computes posterior marginal that can be used later on during the inference procedure. An important part of the functional forms constraint implementation is the [`prod`](@ref) function. More information about [`prod`](@ref) function is present in the [Prod Implementation](@ref lib-prod) section. For example, if we refer to our `CustomFunctionalForm` as to `f` we can see the whole functional form constraints pipeline as:

```math
q(x) = f\left(\frac{\overrightarrow{\mu}(x)\overleftarrow{\mu}(x)}{\int \overrightarrow{\mu}(x)\overleftarrow{\mu}(x) \mathrm{d}x}\right)
```

## Interface

`ReactiveMP.jl`, however, uses some extra utility functions to define functional form constraint behaviour. Here we briefly describe all utility function. If you are only interested in the concrete example, you may directly head to the [Custom Functional Form](@ref custom-functional-form-example) example at the end of this section.

### Abstract super type

```@docs 
AbstractFormConstraint
```
 
### Form check strategy

Every custom functional form must implement a new method for the [`default_form_check_strategy`](@ref) function that returns either [`FormConstraintCheckEach`](@ref) or [`FormConstraintCheckLast`](@ref).

- `FormConstraintCheckLast`: `q(x) = f(μ1(x) * μ2(x) * μ3(x))`
- `FormConstraintCheckEach`: `q(x) = f(f(μ1(x) * μ2(x)) * μ3(x))`

```@docs 
default_form_check_strategy
FormConstraintCheckEach
FormConstraintCheckLast
```

### Prod constraint 

Every custom functional form must implement a new method for the [`default_prod_constraint`](@ref) function that returns a proper `prod_constraint` object.

```@docs 
default_prod_constraint
```

### Constrain form, a.k.a `f`

The main function that a custom functional form must implement, which we referred to as `f` in the beginning of this section, is the [`constrain_form`](@ref) function.

```@docs
constrain_form
```

### Is point mass form constraint (optional)

Every custom functional form may implement a new method for the [`is_point_mass_form_constraint`](@ref) function that returns either `true` or `false`. This is an utility function that simplifes computation of the Bethe Free Energy and is not strictly necessary.

```@docs 
is_point_mass_form_constraint
```

### Compatibility with `@constraints` macro (optional)

To make custom functional form constraint compatible with the `@constraints` macro, it must implement a new method for the [`make_form_constraint`](@ref) function.

```@docs 
make_form_constraint
```

## [Custom Functional Form Example](@id custom-functional-form-example)

In this demo we show how to build a custom functional form constraint that is compatible with the `ReactiveMP.jl` inference backend. An important part of the functional forms constraint implementation is the [`prod`](@ref) function. More information about [`prod`](@ref) function is present in the [Prod Implementation](@ref lib-prod) section. We show a relatively simple use-case, which might not be very useful in practice, but serves as a simple step-by-step guide. Assume that we want a specific posterior marginal of some random variable in our model to have a specific Gaussian parametrisation, for example mean-precision. We can use built-in `NormalMeanPrecision` distribution, but we still need to define our custom functional form constraint:

```@example custom-functional-form-example
using ReactiveMP, GraphPPL

# First we define our functional form structure with no fields
struct MeanPrecisionFormConstraint <: AbstractFormConstraint end
```

Next we define the behaviour of our functional form constraint:

```@example custom-functional-form-example
ReactiveMP.is_point_mass_form_constraint(::MeanPrecisionFormConstraint) = false
ReactiveMP.default_form_check_strategy(::MeanPrecisionFormConstraint)   = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::MeanPrecisionFormConstraint)       = ProdGeneric()

function ReactiveMP.constrain_form(::MeanPrecisionFormConstraint, distribution) 
    # This is quite a naive assumption, that a given `dsitribution` object has `mean` and `precision` defined
    # However this quantities might be approximated with some other external method, e.g. Laplace approximation
    m = mean(distribution)      # or approximate with some other method
    p = precision(distribution) # or approximate with some other method
    return NormalMeanPrecision(m, p)
end

function ReactiveMP.constrain_form(::MeanPrecisionFormConstraint, distribution::DistProduct)
    # DistProduct is the special case, read about this type more in the corresponding documentation section
    # ... 
end
```

At this point we already can use our functional form constraint in the inference backend, however, lets also make our functional form constraint compatible with the `@constraints` macro from `GraphPPL.jl` package.

```@example custom-functional-form-example
ReactiveMP.make_form_constraint(::Type{ NormalMeanPrecision }, args...; kwargs...) = MeanPrecisionFormConstraint()
```

```@example custom-functional-form-example
@constraints begin 
    q(x) :: NormalMeanPrecision
end
```



