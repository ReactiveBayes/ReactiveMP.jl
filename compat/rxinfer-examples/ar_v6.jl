# "Problem Specific/Autoregressive Models": a latent AR(5) with free energy, the sinusoidal AR(2)
# and the AAL stock AR(50) with predictions over missing observations (UnfactorizedData), and the
# ARMA(10, 4) model on the stock data. RxInfer 5.5.2 on ReactiveMP 6.5.0.
using RxInfer
include(joinpath(@__DIR__, "common.jl"))

const SIDE = "v6"
r = Recorder()

ar_unit(order) = ReactiveMP.ar_unit(Multivariate, order)

@model function lar_multivariate(y, order, γ)
    # `c` is a unit vector of size `order` with first element equal to 1
    c = ar_unit(order)
    τ ~ Gamma(α = 1.0, β = 1.0)
    θ ~ MvNormal(mean = zeros(order), precision = diageye(order))
    x0 ~ MvNormal(mean = zeros(order), precision = diageye(order))
    x_prev = x0
    for i in eachindex(y)
        x[i] ~ AR(x_prev, θ, τ)
        y[i] ~ Normal(mean = dot(c, x[i]), precision = γ)
        x_prev = x[i]
    end
end

@constraints function ar_constraints()
    q(x0, x, θ, τ) = q(x0, x)q(θ)q(τ)
end

@meta function ar_meta(order)
    AR() -> ARMeta(Multivariate, order, ARsafe())
end

@initialization function ar_init(order)
    q(τ) = GammaShapeRate(1.0, 1.0)
    q(θ) = MvNormalMeanPrecision(zeros(order), diageye(order))
end

function record_ar!(r, prefix, result; full_x = true)
    # For the AR(50) the states' 50×50 covariances would make the file 26 MB; their means and
    # variances, what the notebook plots, are kept instead.
    xs = result.posteriors[:x]
    record!(r, "$prefix:x", full_x ? xs : [vcat(mean(d), var(d)) for d in xs])
    record!(r, "$prefix:τ", result.posteriors[:τ])
    record!(r, "$prefix:θ", result.posteriors[:θ])
    return r
end

# --- AR(5) -----------------------------------------------------------------------------------

real_θ = coefs_ar_5
real_τ = 0.5
real_γ = 2.0
order = length(real_θ)
n = 500
states, observations = ar_generate_synthetic_dataset(n = n, θ = real_θ, τ = real_τ, γ = real_γ)
record!(r, "data:ar5", observations)

result = infer(
    model = lar_multivariate(order = order, γ = real_γ),
    data = (y = observations,),
    constraints = ar_constraints(),
    meta = ar_meta(order),
    initialization = ar_init(order),
    options = (limit_stack_depth = 500,),
    returnvars = (x = KeepLast(), τ = KeepEach(), θ = KeepEach()),
    free_energy = true,
    iterations = 20
)
record_ar!(r, "ar5", result)
record!(r, "ar5:free_energy", result.free_energy)

# --- sinusoidal AR(2), predictions -----------------------------------------------------------

predictions_coefficients = ar_generate_sinusoidal_coefficients(f = 0.03)
predictions_dataset = ar_generate_synthetic_dataset(n = 350, θ = predictions_coefficients, τ = 1.0, γ = 0.01)
number_of_predictions = 100
predictions_states, predictions_observations = predictions_dataset
record!(r, "data:sin", predictions_observations)
predictions_observations_with_predictions = vcat(predictions_observations, [missing for _ in 1:number_of_predictions])

predictions_result = infer(
    model = lar_multivariate(order = 2, γ = 0.01),
    data = (y = UnfactorizedData(predictions_observations_with_predictions),),
    constraints = ar_constraints(),
    meta = ar_meta(2),
    initialization = ar_init(2),
    options = (limit_stack_depth = 500,),
    returnvars = (x = KeepLast(), τ = KeepEach(), θ = KeepEach()),
    free_energy = false,
    iterations = 20
)
record_ar!(r, "sin", predictions_result)
record!(r, "sin:y_predictions", predictions_result.predictions[:y][end])

# --- AAL stock, AR(50), predictions ----------------------------------------------------------

x_data = ar_stock_data()
record!(r, "data:stock", x_data)
observed_size = length(x_data) - 50
x_observed = Float64.(x_data[1:observed_size])
x_to_predict = Float64.(x_data[(observed_size + 1):end])
stock_observations_with_predictions = vcat(x_observed, [missing for _ in 1:length(x_to_predict)])

stock_predictions_result = infer(
    model = lar_multivariate(order = 50, γ = 1.0),
    data = (y = UnfactorizedData(stock_observations_with_predictions),),
    constraints = ar_constraints(),
    meta = ar_meta(50),
    initialization = ar_init(50),
    options = (limit_stack_depth = 500,),
    returnvars = (x = KeepLast(), τ = KeepEach(), θ = KeepEach()),
    free_energy = false,
    iterations = 20
)
record_ar!(r, "stock", stock_predictions_result; full_x = false)
record!(r, "stock:y_predictions", stock_predictions_result.predictions[:y][end])

# --- ARMA(10, 4) -----------------------------------------------------------------------------

@model function ARMA(x, x_prev, priors, p_order, q_order)
    c = zeros(q_order); c[1] = 1.0
    S = ar_shift(q_order) # MA
    γ ~ priors[:γ]
    η ~ priors[:η]
    θ ~ priors[:θ]
    τ ~ priors[:τ]
    h[1] ~ priors[:h]
    z[1] ~ AR(h[1], η, τ)
    e[1] ~ Normal(mean = 0.0, precision = γ)
    x[1] ~ dot(c, z[1]) + dot(θ, x_prev[1]) + e[1]
    for t in 1:(length(x) - 1)
        h[t + 1] ~ S * h[t] + c * e[t]
        z[t + 1] ~ AR(h[t + 1], η, τ)
        e[t + 1] ~ Normal(mean = 0.0, precision = γ)
        x[t + 1] ~ dot(c, z[t + 1]) + dot(θ, x_prev[t + 1]) + e[t + 1]
    end
end

@constraints function arma_constraints()
    q(z, h, η, τ, γ, e) = q(z, h)q(η)q(τ)q(γ)q(e)
end

@initialization function arma_initialization(priors)
    q(h) = priors[:h]
    μ(h) = priors[:h]
    q(γ) = priors[:γ]
    q(τ) = priors[:τ]
    q(η) = priors[:η]
    q(θ) = priors[:θ]
end

p_order = 10 # AR
q_order = 4  # MA
priors = (
    h = MvNormalMeanPrecision(zeros(q_order), diageye(q_order)),
    γ = GammaShapeRate(1.0e4, 1.0),
    τ = GammaShapeRate(1.0e2, 1.0),
    η = MvNormalMeanPrecision(ones(q_order), diageye(q_order)),
    θ = MvNormalMeanPrecision(zeros(p_order), diageye(p_order)),
)

arma_x_data = Float64.(x_data[(p_order + 1):end])[1:observed_size]
arma_x_prev_data = [Float64.(x_data[(i + p_order - 1):-1:i]) for i in 1:(length(x_data) - p_order)][1:observed_size]

arma_result = infer(
    model = ARMA(priors = priors, p_order = p_order, q_order = q_order),
    data = (x = arma_x_data, x_prev = arma_x_prev_data),
    initialization = arma_initialization(priors),
    constraints = arma_constraints(),
    meta = ar_meta(q_order),
    returnvars = KeepLast(),
    iterations = 20,
    options = (limit_stack_depth = 400,),
)
for v in sort(collect(keys(arma_result.posteriors)))
    record!(r, "arma:$v", arma_result.posteriors[v])
end

save_results(r, "ar", SIDE)
