# "Basic Examples/Kalman filtering and smoothing", its nonlinear parts: the identification
# problem with `s[i] := f(x[i], w[i])` for `f = +` (the addition node) and for `f = smooth_min`
# (a Delta node under Linearization), and the streaming (online) version with `smooth_min`.
# RxInfer 5.5.2 on ReactiveMP 6.5.0.
using RxInfer
include(joinpath(@__DIR__, "common.jl"))

const SIDE = "v6"
r = Recorder()
seed = 1234

@model function identification_problem(f, y, m_x_0, τ_x_0, a_x, b_x, m_w_0, τ_w_0, a_w, b_w, a_y, b_y)
    x0 ~ Normal(mean = m_x_0, precision = τ_x_0)
    τ_x ~ Gamma(shape = a_x, rate = b_x)
    w0 ~ Normal(mean = m_w_0, precision = τ_w_0)
    τ_w ~ Gamma(shape = a_w, rate = b_w)
    τ_y ~ Gamma(shape = a_y, rate = b_y)
    x_i_min = x0
    w_i_min = w0
    local x
    local w
    local s
    for i in 1:length(y)
        x[i] ~ Normal(mean = x_i_min, precision = τ_x)
        w[i] ~ Normal(mean = w_i_min, precision = τ_w)
        s[i] := f(x[i], w[i])
        y[i] ~ Normal(mean = s[i], precision = τ_y)
        x_i_min = x[i]
        w_i_min = w[i]
    end
end

constraints = @constraints begin
    q(x0, w0, x, w, τ_x, τ_w, τ_y, s) = q(x, x0, w, w0, s)q(τ_w)q(τ_x)q(τ_y)
end

function record_identification!(r, prefix, result)
    for v in (:τ_x, :τ_w, :τ_y)
        record!(r, "$prefix:$v", result.posteriors[v])
    end
    # The notebook plots the last iteration of `x`, `w` and `s`.
    for v in (:x, :w, :s)
        record!(r, "$prefix:$v", result.posteriors[v][end])
    end
    return r
end

# --- f = + -----------------------------------------------------------------------------------

n = 250
real_x, real_w, real_y = identification_generate_data(+, n)
record!(r, "data:plus_y", real_y)

m_x_0, τ_x_0 = -20.0, 1.0
m_w_0, τ_w_0 = 20.0, 1.0
a_x, b_x = 0.01, 0.01var(real_x)
a_w, b_w = 0.01, 0.01var(real_w)
a_y, b_y = 1.0, 1.0

xinit = map(r -> NormalMeanPrecision(r, τ_x_0), reverse(range(-60, -20, length = n)))
winit = map(r -> NormalMeanPrecision(r, τ_w_0), range(20, 60, length = n))

init = @initialization begin
    μ(x) = xinit
    μ(w) = winit
    q(τ_x) = GammaShapeRate(a_x, b_x)
    q(τ_w) = GammaShapeRate(a_w, b_w)
    q(τ_y) = GammaShapeRate(a_y, b_y)
end

result = infer(
    model = identification_problem(f = +, m_x_0 = m_x_0, τ_x_0 = τ_x_0, a_x = a_x, b_x = b_x, m_w_0 = m_w_0, τ_w_0 = τ_w_0, a_w = a_w, b_w = b_w, a_y = a_y, b_y = b_y),
    data = (y = real_y,),
    options = (limit_stack_depth = 500,),
    constraints = constraints,
    initialization = init,
    iterations = 50,
    # Not in the notebook: the free energy, to check the Gamma priors' energies.
    free_energy = true
)
record_identification!(r, "plus", result)
record!(r, "plus:free_energy", result.free_energy)

# --- f = smooth_min, a Delta node under Linearization --------------------------------------------

min_meta = @meta begin
    smooth_min() -> Linearization()
end

n = 200
min_real_x, min_real_w, min_real_y = identification_generate_data(min, n, seed = seed, x_i_min = 0.0, w_i_min = 0.0, noise = 1.0, real_x_τ = 1.0, real_w_τ = 1.0)
record!(r, "data:min_y", min_real_y)

min_m_x_0, min_τ_x_0 = -1.0, 1.0
min_m_w_0, min_τ_w_0 = 1.0, 1.0
min_a_x, min_b_x = 1.0, 1.0
min_a_w, min_b_w = 1.0, 1.0
min_a_y, min_b_y = 1.0, 1.0

init = @initialization begin
    μ(x) = NormalMeanPrecision(min_m_x_0, min_τ_x_0)
    μ(w) = NormalMeanPrecision(min_m_w_0, min_τ_w_0)
    q(τ_x) = GammaShapeRate(min_a_x, min_b_x)
    q(τ_w) = GammaShapeRate(min_a_w, min_b_w)
    q(τ_y) = GammaShapeRate(min_a_y, min_b_y)
end

min_result = infer(
    model = identification_problem(f = smooth_min, m_x_0 = min_m_x_0, τ_x_0 = min_τ_x_0, a_x = min_a_x, b_x = min_b_x, m_w_0 = min_m_w_0, τ_w_0 = min_τ_w_0, a_w = min_a_w, b_w = min_b_w, a_y = min_a_y, b_y = min_b_y),
    data = (y = min_real_y,),
    options = (limit_stack_depth = 500,),
    constraints = constraints,
    initialization = init,
    meta = min_meta,
    iterations = 50,
    free_energy = true
)
record_identification!(r, "min", min_result)
record!(r, "min:free_energy", min_result.free_energy)

# --- streaming, f = smooth_min -------------------------------------------------------------------

@model function rx_identification(f, m_x_0, τ_x_0, m_w_0, τ_w_0, a_x, b_x, a_y, b_y, a_w, b_w, y)
    x0 ~ Normal(mean = m_x_0, precision = τ_x_0)
    τ_x ~ Gamma(shape = a_x, rate = b_x)
    w0 ~ Normal(mean = m_w_0, precision = τ_w_0)
    τ_w ~ Gamma(shape = a_w, rate = b_w)
    τ_y ~ Gamma(shape = a_y, rate = b_y)
    x ~ Normal(mean = x0, precision = τ_x)
    w ~ Normal(mean = w0, precision = τ_w)
    s := f(x, w)
    y ~ Normal(mean = s, precision = τ_y)
end

rx_constraints = @constraints begin
    q(x0, x, w0, w, τ_x, τ_w, τ_y, s) = q(x0, x)q(w, w0)q(τ_w)q(τ_x)q(s)q(τ_y)
end

autoupdates = @autoupdates begin
    m_x_0, τ_x_0 = mean_precision(q(x))
    m_w_0, τ_w_0 = mean_precision(q(w))
    a_x = shape(q(τ_x))
    b_x = rate(q(τ_x))
    a_y = shape(q(τ_y))
    b_y = rate(q(τ_y))
    a_w = shape(q(τ_w))
    b_w = rate(q(τ_w))
end

rx_meta = @meta begin
    smooth_min() -> Linearization()
end

n = 300
rx_real_x, rx_real_w, rx_real_y = identification_generate_data(min, n, seed = seed, x_i_min = 1.0, w_i_min = -1.0, noise = 1.0, real_x_τ = 1.0, real_w_τ = 1.0)
record!(r, "data:rx_y", rx_real_y)

init = @initialization begin
    q(w) = NormalMeanVariance(-2.0, 1.0)
    q(x) = NormalMeanVariance(2.0, 1.0)
    q(τ_x) = GammaShapeRate(1.0, 1.0)
    q(τ_w) = GammaShapeRate(1.0, 1.0)
    q(τ_y) = GammaShapeRate(1.0, 20.0)
end

engine = infer(
    model = rx_identification(f = smooth_min),
    constraints = rx_constraints,
    data = (y = rx_real_y,),
    autoupdates = autoupdates,
    meta = rx_meta,
    returnvars = (:x, :w, :τ_x, :τ_w, :τ_y, :s),
    keephistory = 1000,
    historyvars = KeepLast(),
    initialization = init,
    iterations = 10,
    free_energy = true,
    free_energy_diagnostics = nothing,
    autostart = true,
)
for v in (:x, :w, :s, :τ_x, :τ_w, :τ_y)
    record!(r, "rx:$v", engine.history[v])
end
record!(r, "rx:free_energy", engine.free_energy_raw_history)

save_results(r, "delta", SIDE)
