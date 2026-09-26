# [Form constraints](@id custom-functional-form)

A form constraint gives the product of the messages at a variable a chosen functional form: it approximates the product of colliding messages by a distribution of that form, which is the marginal the rest of inference uses. It is how a model keeps a posterior tractable where the product has no closed form, or keeps it in a parameterisation it needs. The product itself is the `prod` function of the [`BayesBase`](https://reactivebayes.github.io/BayesBase.jl/stable/) package. With the constraint as `f`, the marginal is:

```math
q(x) = f\left(\frac{\overrightarrow{\mu}(x)\overleftarrow{\mu}(x)}{\int \overrightarrow{\mu}(x)\overleftarrow{\mu}(x) \mathrm{d}x}\right)
```

## [The interface](@id custom-functional-form-interface)

A form constraint is given to a variable through its [`ReactiveMP.MessageProductContext`](@ref)s (`form_constraint`, `form_constraint_check_strategy` and `prod_constraint`); RxInfer builds these from a model's constraints. A constraint implements the functions below. For a complete constraint, see the [example](@ref custom-functional-form-example) at the end of this page.

### The supertype

```@docs 
AbstractFormConstraint
UnspecifiedFormConstraint
CompositeFormConstraint
ReactiveMP.preprocess_form_constraints
```
 
### Form check strategy

A form constraint implements [`default_form_check_strategy`](@ref), which returns [`FormConstraintCheckEach`](@ref) or [`FormConstraintCheckLast`](@ref):

- `FormConstraintCheckLast`: `q(x) = f(μ1(x) * μ2(x) * μ3(x))`
- `FormConstraintCheckEach`: `q(x) = f(f(μ1(x) * μ2(x)) * μ3(x))`

```@docs 
default_form_check_strategy
FormConstraintCheckEach
FormConstraintCheckLast
FormConstraintCheckPickDefault
```

### The product strategy

A form constraint implements [`default_prod_constraint`](@ref), the product strategy it needs from `BayesBase.prod`.

```@docs 
default_prod_constraint
```

### The constraint, `f`

The main function a form constraint implements, the `f` above, is [`constrain_form`](@ref). A constraint that changes the product, returning something other than it was given, leaves the product's log scale undefined.

```@docs
constrain_form
```

## [An example](@id custom-functional-form-example)

This example builds a form constraint that keeps a marginal in the mean-precision parameterisation of a normal, `MeanPrecisionFormConstraint`: a simple use of the interface, step by step.

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

## [Wrapped form constraints](@id custom-functional-form-wrapped)

A constraint need not subtype `AbstractFormConstraint`, as when it is defined in another package or subtypes another abstract type. [`ReactiveMP.preprocess_form_constraints`](@ref) wraps such an object in a [`ReactiveMP.WrappedFormConstraint`](@ref), which makes it a form constraint.

A wrapped constraint may implement [`ReactiveMP.prepare_context`](@ref): its result is stored with the constraint, computed once, and [`constrain_form`](@ref) is then called with three arguments, the constraint, the context and the distribution.

```@docs 
ReactiveMP.WrappedFormConstraint
ReactiveMP.prepare_context
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


