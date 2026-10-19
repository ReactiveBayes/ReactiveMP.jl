# [Form constraints](@id custom-functional-form)

A form constraint gives a variable's marginal a chosen functional form. The variable multiplies
the messages that arrive at it, and the constraint approximates their normalised product by a
distribution of that form. With the constraint as ``f``, the marginal is

```math
q(x) = f\left(\frac{\overrightarrow{\mu}(x)\overleftarrow{\mu}(x)}{\int \overrightarrow{\mu}(x)\overleftarrow{\mu}(x) \mathrm{d}x}\right).
```

A form constraint keeps a posterior tractable where the product has no closed form, or keeps it
in a parameterisation the model needs. The product itself is the `prod` function of
[BayesBase](https://reactivebayes.github.io/BayesBase.jl/stable/).

```@setup form
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, Rocket, Random
import ReactiveMP: activate!, FactorNodeActivationOptions, MessageProductContext, get_stream_of_marginals
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
```

The examples use the `Gaussian` node of [The example node](@ref example-node).

## [The interface](@id custom-functional-form-interface)

A variable receives its form constraint through its [`ReactiveMP.MessageProductContext`](@ref)s,
as the keywords `form_constraint`, `form_constraint_check_strategy` and `prod_constraint`. RxInfer
builds them from a model's constraints. A constraint implements the functions below, and the
[example](@ref custom-functional-form-example) implements all of them.

### The supertype

A form constraint subtypes [`AbstractFormConstraint`](@ref).
[`UnspecifiedFormConstraint`](@ref), the default, leaves the product as it is, and
[`CompositeFormConstraint`](@ref) applies several constraints in turn.

```@docs
AbstractFormConstraint
UnspecifiedFormConstraint
CompositeFormConstraint
ReactiveMP.preprocess_form_constraints
```

### Form check strategy

A form constraint implements [`default_form_check_strategy`](@ref), which says when it applies:

- [`FormConstraintCheckLast`](@ref) applies it once, to the whole product:
  ``q(x) = f(\mu_1(x) \mu_2(x) \mu_3(x))``;
- [`FormConstraintCheckEach`](@ref) applies it after each pairwise product:
  ``q(x) = f(f(\mu_1(x) \mu_2(x)) \mu_3(x))``.

```@docs
default_form_check_strategy
FormConstraintCheckEach
FormConstraintCheckLast
FormConstraintCheckPickDefault
```

### The product strategy

A form constraint implements [`default_prod_constraint`](@ref), the strategy it needs
`BayesBase.prod` to multiply with. `GenericProd()` computes a closed-form product where one
exists, and otherwise keeps the two sides as a `BayesBase.ProductOf`.

```@docs
default_prod_constraint
```

### The constraint

[`constrain_form`](@ref) is the ``f`` above: it takes the product and returns a distribution of
the chosen form. A constraint that changes the product, returning something other than it was
given, leaves the product's [log scale](@ref lib-logscale) undefined.

```@docs
constrain_form
```

## [An example](@id custom-functional-form-example)

The constraint below keeps a marginal a normal distribution in the mean-precision
parameterisation:

```@example form
struct MeanPrecisionFormConstraint <: AbstractFormConstraint end

ReactiveMP.default_form_check_strategy(::MeanPrecisionFormConstraint) = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::MeanPrecisionFormConstraint) = GenericProd()

ReactiveMP.constrain_form(::MeanPrecisionFormConstraint, distribution) =
    NormalMeanPrecision(mean(distribution), precision(distribution))

constraint = ReactiveMP.preprocess_form_constraints(MeanPrecisionFormConstraint())
constrain_form(constraint, NormalMeanVariance(0.0, 2.0))
```

The first method needs a distribution with a mean and a precision. A product with no closed form
is a `ProductOf`, which has neither. The second method matches its moments on a grid, using the
product's unnormalised log-density:

```@example form
function ReactiveMP.constrain_form(::MeanPrecisionFormConstraint, product::ProductOf)
    left = product.left
    xs = range(mean(left) - 10 * std(left), mean(left) + 10 * std(left); length = 2001)
    weights = exp.(logpdf.(Ref(product), xs))
    weights ./= sum(weights)
    m = sum(weights .* xs)
    v = sum(weights .* abs2.(xs .- m))
    return NormalMeanPrecision(m, inv(v))
end

constrain_form(constraint, prod(GenericProd(), NormalMeanVariance(0.0, 1.0), Laplace(1.0, 0.5)))
```

The grid spans ten standard deviations of the left side, so the method assumes the left side
covers the product's mass.

A variable applies the constraint to its marginal when its marginal's product context carries it:

```@example form
c = MeanPrecisionFormConstraint()
marginal_context = MessageProductContext(;
    form_constraint = c,
    prod_constraint = default_prod_constraint(c),
    form_constraint_check_strategy = default_form_check_strategy(c),
)

x, y = randomvar(label = :x), datavar(label = :y)
prior = factornode(Gaussian, [(:out, x), (:μ, constvar(0.0)), (:v, constvar(10.0))])
likelihood = factornode(Gaussian, [(:out, y), (:μ, x), (:v, constvar(1.0))])
activate!(x, RandomVariableActivationOptions(nothing, MessageProductContext(), marginal_context))
activate!(y, DataVariableActivationOptions())
foreach(n -> activate!(n, FactorNodeActivationOptions(; logscales = true)), (prior, likelihood))

subscription = subscribe!(get_stream_of_marginals(x), (q) -> println(q))
new_observation!(y, 2.0)
unsubscribe!(subscription)
```

The posterior is a `NormalMeanPrecision`. The constraint changed the product's type, so its log
scale is undefined, and says it was the form constraint.

## [Wrapped form constraints](@id custom-functional-form-wrapped)

A constraint need not subtype `AbstractFormConstraint`, as when another package defines it or it
subtypes another abstract type. [`ReactiveMP.preprocess_form_constraints`](@ref) wraps such an
object in a [`ReactiveMP.WrappedFormConstraint`](@ref), which makes it a form constraint.

A wrapped constraint may implement [`ReactiveMP.prepare_context`](@ref). The engine computes its
result once and stores it with the constraint. [`constrain_form`](@ref) then receives three
arguments: the constraint, the context and the distribution.

```@docs
ReactiveMP.WrappedFormConstraint
ReactiveMP.prepare_context
```

The constraint below replaces a distribution by ten samples from it, drawn with the random number
generator its context holds:

```@example form
struct SampleListFormConstraint end

ReactiveMP.default_form_check_strategy(::SampleListFormConstraint) = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::SampleListFormConstraint) = GenericProd()
ReactiveMP.prepare_context(::SampleListFormConstraint) = Xoshiro(42)
ReactiveMP.constrain_form(::SampleListFormConstraint, rng, distribution) = rand(rng, distribution, 10)

sampler = ReactiveMP.preprocess_form_constraints(SampleListFormConstraint())
constrain_form(sampler, Normal(0.0, 10.0))
```
