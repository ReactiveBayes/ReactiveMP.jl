# [Custom Functional Form Specification](@id custom-functional-form)

In a nutshell, functional form constraints defines a function that approximates the product of colliding messages and computes posterior marginal that can be used later on during the inference procedure. An important part of the functional forms constraint implementation is the `prod` function in the [`BayesBase`](https://reactivebayes.github.io/BayesBase.jl/stable/) package. For example, if we refer to our `CustomFunctionalForm` as to `f` we can see the whole functional form constraints pipeline as:

```math
q(x) = f\left(\frac{\overrightarrow{\mu}(x)\overleftarrow{\mu}(x)}{\int \overrightarrow{\mu}(x)\overleftarrow{\mu}(x) \mathrm{d}x}\right)
```

## Interface

`ReactiveMP.jl`, however, uses some extra utility functions to define functional form constraint behaviour. Here we briefly describe all utility function. If you are only interested in the concrete example, you may directly head to the [Custom Functional Form](@ref custom-functional-form-example) example at the end of this section.

### Abstract super type

```@docs 
AbstractFormConstraint
UnspecifiedFormConstraint
CompositeFormConstraint
ReactiveMP.preprocess_form_constraints
```
 
### Form check strategy

Every custom functional form must implement a new method for the [`default_form_check_strategy`](@ref) function that returns either [`FormConstraintCheckEach`](@ref) or [`FormConstraintCheckLast`](@ref).

- `FormConstraintCheckLast`: `q(x) = f(μ1(x) * μ2(x) * μ3(x))`
- `FormConstraintCheckEach`: `q(x) = f(f(μ1(x) * μ2(x)) * μ3(x))`

```@docs 
default_form_check_strategy
FormConstraintCheckEach
FormConstraintCheckLast
FormConstraintCheckPickDefault
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

## [Custom Functional Form Example](@id custom-functional-form-example)

In this demo, we show how to build a custom functional form constraint that is compatible with the `ReactiveMP.jl` inference backend. An important part of the functional form constraint implementation is the `prod` function in the [`BayesBase`](https://reactivebayes.github.io/BayesBase.jl/stable/) package. We present a relatively simple use case, which may not be very practical but serves as a straightforward step-by-step guide.

Assume that we want a specific posterior marginal of some random variable in our model to have a specific Gaussian parameterization, such as mean-precision. Here, how we can achieve this with our custom `MeanPrecisionFormConstraint` functional form constraint:

```@example custom-functional-form-example
using ReactiveMP, ExponentialFamily, Distributions, BayesBase

# First, we define our functional form structure with no fields
struct MeanPrecisionFormConstraint <: AbstractFormConstraint end

ReactiveMP.default_form_check_strategy(::MeanPrecisionFormConstraint) = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::MeanPrecisionFormConstraint) = GenericProd()

function ReactiveMP.constrain_form(::MeanPrecisionFormConstraint, distribution) 
    # This assumes that the given `distribution` object has `mean` and `precision` defined.
    # These quantities might be approximated using other methods, such as Laplace approximation.
    m = mean(distribution)      # or approximate with some other method
    p = precision(distribution) # or approximate with some other method
    return NormalMeanPrecision(m, p)
end

function ReactiveMP.constrain_form(::MeanPrecisionFormConstraint, distribution::BayesBase.ProductOf)
    # `ProductOf` is a special case. Read more about this type in the corresponding 
    # documentation section of the `BayesBase` package.
    # ... 
end

constraint = ReactiveMP.preprocess_form_constraints(MeanPrecisionFormConstraint())

constrain_form(constraint, NormalMeanVariance(0, 2))
```

## Wrapped Form Constraints 

Some constraint objects might not be subtypes of `AbstractFormConstraint`. This can occur, for instance, if the object is defined in a different package or needs to subtype a different abstract type. In such cases, `ReactiveMP` expects users to pass a `WrappedFormConstraint` object, which wraps the original object and makes it compatible with the `ReactiveMP` inference backend. Note that the [`ReactiveMP.preprocess_form_constraints`](@ref) function automatically wraps all objects that are not subtypes of `AbstractFormConstraint`.

Additionally, objects wrapped by `WrappedFormConstraints` may implement the `ReactiveMP.prepare_context` function. This function's output will be stored in the `WrappedFormConstraints` along with the original object. If `prepare_context` is implemented, the `constrain_form` function will take three arguments: the original constraint, the context, and the object that needs to be constrained.

```@docs 
ReactiveMP.WrappedFormConstraint
ReactiveMP.prepare_context
ReactiveMP.constrain_form(::ReactiveMP.WrappedFormConstraint, something)
```

```@example wrapped-form-constraint-example
using ReactiveMP, Distributions, BayesBase, Random

# First, we define our custom form constraint that creates a set of samples
# Note that this is not a subtype of `AbstractFormConstraint`
struct MyCustomSampleListFormConstraint end

# Note that we still need to implement `default_form_check_strategy` and `default_prod_constraint` functions
#  which are necessary for the `ReactiveMP` inference backend
ReactiveMP.default_form_check_strategy(::MyCustomSampleListFormConstraint) = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::MyCustomSampleListFormConstraint) = GenericProd()

# We implement the `prepare_context` function, which returns a random number generator
function ReactiveMP.prepare_context(constraint::MyCustomSampleListFormConstraint)
    return Random.default_rng()
end

# We implement the `constrain_form` function, which returns a set of samples
function ReactiveMP.constrain_form(constraint::MyCustomSampleListFormConstraint, context, distribution)
    return rand(context, distribution, 10)
end

constraint = ReactiveMP.preprocess_form_constraints(MyCustomSampleListFormConstraint())

constrain_form(constraint, Normal(0, 10))
```


