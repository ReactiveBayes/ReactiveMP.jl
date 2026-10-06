# Choosing a method

Each method answers one question: if `x` is normal with mean `m` and covariance `V`, what are the
mean and covariance of `f(x)`? For an affine `f` the answer is exact and normal. For any other
`f`, the distribution of `f(x)` is not normal, and each method approximates its moments in a
different way.

| method | needs | returns | suits |
|---|---|---|---|
| [`Linearization`](@ref) | `f` differentiable by ForwardDiff | the local linear map `(A, b)` | functions close to linear over the inputs' spread; exact for affine ones |
| [`Unscented`](@ref) | `f` evaluated at `2d + 1` points | the output's mean and covariance | any function, no derivatives; accurate to second order |
| [`GaussHermiteCubature`](@ref) | `f` evaluated at `p^d` points | weights and points, for any expectation | expectations and reweighted moments in a few dimensions |

## One function, every method

The comparison uses a position measured in polar coordinates and converted to Cartesian ones. The
radius `r` is known to within `0.1` and the angle `θ` to within about `0.55` radians:

```@example choosing
using MessagePassingRulesApproximations, LinearAlgebra, Random

polar_to_cartesian(x) = [x[1] * cos(x[2]), x[1] * sin(x[2])]

m, V = [1.0, 0.0], [0.01 0.0; 0.0 0.3]
nothing # hide
```

The radius and the angle are independent, so the exact moments are known. With
``s`` the angle's variance, ``\mathrm{E}[\cos θ] = e^{-s/2}``,
``\mathrm{E}[\cos^2 θ] = (1 + e^{-2s}) / 2`` and ``\mathrm{E}[\sin^2 θ] = (1 - e^{-2s}) / 2``,
while ``\mathrm{E}[\sin θ \cos θ] = 0``:

```@example choosing
r2, s = m[1]^2 + V[1, 1], V[2, 2]   # E[r²] and the angle's variance

exact_mean = [m[1] * exp(-s / 2), 0.0]
exact_cov = [r2 * (1 + exp(-2s)) / 2 - exact_mean[1]^2 0.0; 0.0 r2 * (1 - exp(-2s)) / 2]
```

Each method below returns the output's mean and covariance, and how many times it evaluated `f`:

```@example choosing
# `f`, and a counter of its calls.
function counting(f)
    calls = Ref(0)
    return (x -> (calls[] += 1; f(x))), calls
end

function linearised_moments(f)
    g, calls = counting(f)
    A, b = approximate(Linearization(), g, (m,))
    return A * m + b, A * V * A', calls[]
end

function unscented_moments(f)
    g, calls = counting(f)
    μ, Σ = approximate(Unscented(), g, (m,), (V,))
    return μ, Σ, calls[]
end

function cubature_moments(f, p)
    g, calls = counting(f)
    gh = ghcubature(p)
    weights = collect(getweights(gh, m, V))
    values = [g(x) for x in getpoints(gh, m, V)]   # each point is used before the next one replaces it
    μ = sum(weights .* values)
    return μ, sum(w * (y - μ) * (y - μ)' for (w, y) in zip(weights, values)), calls[]
end

function monte_carlo_moments(f, n)
    g, calls = counting(f)
    rng, L = Xoshiro(1), cholesky(V).L
    values = [g(m + L * randn(rng, 2)) for _ in 1:n]
    μ = sum(values) / n
    return μ, sum((y - μ) * (y - μ)' for y in values) / (n - 1), calls[]
end
nothing # hide
```

[`getpoints`](@ref) yields one buffer for every multivariate point, so the cubature evaluates
each point as it arrives. The errors are distances to the exact moments:

```@example choosing
results = [
    "linearisation" => linearised_moments(polar_to_cartesian),
    "unscented" => unscented_moments(polar_to_cartesian),
    "Gauss–Hermite, 3 points" => cubature_moments(polar_to_cartesian, 3),
    "Gauss–Hermite, 5 points" => cubature_moments(polar_to_cartesian, 5),
    "Gauss–Hermite, 10 points" => cubature_moments(polar_to_cartesian, 10),
    "Monte Carlo, 10⁵ samples" => monte_carlo_moments(polar_to_cartesian, 100_000),
]

println(rpad("method", 26), lpad("calls", 8), lpad("mean error", 14), lpad("cov. error", 14))
for (name, (μ, Σ, calls)) in results
    mean_error = round(norm(μ - exact_mean); sigdigits = 2)
    cov_error = round(norm(Σ - exact_cov); sigdigits = 2)
    println(rpad(name, 26), lpad(calls, 8), lpad(mean_error, 14), lpad(cov_error, 14))
end
```

## Reading the comparison

- **Linearisation** is the cheapest: ForwardDiff evaluates `f` once with dual numbers, besides
  the value at the mean. It keeps no curvature, so the mean stays at the measured radius, `1`,
  and the covariance is the input's. It suits a function that is close to linear over the
  inputs' spread.
- **The unscented transform** evaluates `f` at `2d + 1` points and needs no derivative. It moves
  the mean towards the origin, following the circle's curvature, and cuts the error of the mean
  more than tenfold. With an angle this uncertain, its covariance is still about as far off as
  the linearisation's: second-order accuracy is not enough here. It is the usual choice for a
  moderate nonlinearity.
- **Gauss–Hermite cubature** converges fast for a smooth `f`: ten points per dimension reach the
  exact moments to rounding. Its cost, `p^d` evaluations, grows exponentially with the dimension
  `d`, so it suits one or a few dimensions.
- **Monte Carlo** works for any `f` and any dimension, but its error falls only as one over the
  square root of the number of samples. A hundred thousand evaluations are less accurate here
  than twenty-five cubature points.

The dimension decides between the last three. In six dimensions the unscented transform takes
13 evaluations, while ten cubature points per dimension take a million.
