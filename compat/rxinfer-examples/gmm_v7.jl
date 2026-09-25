# "Problem Specific/Gaussian Mixture": a univariate two-component NormalMixture (Beta, Normal and
# Gamma priors) and a bivariate six-component one (MvNormal and Wishart priors, a Dirichlet on the
# weights), both mean-field, with free energy. RxInfer v7 branch on ReactiveMP v7; nothing to port.
using RxInfer
include(joinpath(@__DIR__, "common.jl"))

const SIDE = "v7"
r = Recorder()

# --- univariate ------------------------------------------------------------------------------

data_univariate = gmm_generate_univariate_data(100)
record!(r, "data:univariate", data_univariate)

@model function univariate_gaussian_mixture_model(y)
    s ~ Beta(1.0, 1.0)
    m[1] ~ Normal(mean = -2.0, variance = 1.0e3)
    w[1] ~ Gamma(shape = 0.01, rate = 0.01)
    m[2] ~ Normal(mean = 2.0, variance = 1.0e3)
    w[2] ~ Gamma(shape = 0.01, rate = 0.01)
    for i in eachindex(y)
        z[i] ~ Bernoulli(s)
        y[i] ~ NormalMixture(switch = z[i], m = m, p = w)
    end
end

n_iterations = 10

init = @initialization begin
    q(s) = vague(Beta)
    q(m) = [NormalMeanVariance(-2.0, 1.0e3), NormalMeanVariance(2.0, 1.0e3)]
    q(w) = [vague(GammaShapeRate), vague(GammaShapeRate)]
end

results_univariate = infer(
    model = univariate_gaussian_mixture_model(),
    constraints = MeanField(),
    data = (y = data_univariate,),
    initialization = init,
    iterations = n_iterations,
    free_energy = true
)
record!(r, "uni:m", results_univariate.posteriors[:m])
record!(r, "uni:w", results_univariate.posteriors[:w])
record!(r, "uni:s", results_univariate.posteriors[:s])
record!(r, "uni:z", results_univariate.posteriors[:z][end])
record!(r, "uni:free_energy", results_univariate.free_energy)

# --- multivariate ----------------------------------------------------------------------------

data_multivariate = gmm_generate_multivariate_data(500)
record!(r, "data:multivariate", reduce(vcat, data_multivariate))

@model function multivariate_gaussian_mixture_model(nr_mixtures, priors, y)
    local m
    local w
    for k in 1:nr_mixtures
        m[k] ~ priors[k]
        w[k] ~ Wishart(3, 1.0e2 * diagm(ones(2)))
    end
    s ~ Dirichlet(ones(nr_mixtures))
    for n in eachindex(y)
        z[n] ~ Categorical(s)
        y[n] ~ NormalMixture(switch = z[n], m = m, p = w)
    end
end

rng = MersenneTwister(121) # unused, as in the notebook
priors = [MvNormal([cos(k * 2π / 6), sin(k * 2π / 6)], diagm(1.0e2 * ones(2))) for k in 1:6]
init = @initialization begin
    q(s) = vague(Dirichlet, 6)
    q(m) = priors
    q(w) = Wishart(3, diagm(1.0e2 * ones(2)))
end

results_multivariate = infer(
    model = multivariate_gaussian_mixture_model(
        nr_mixtures = 6,
        priors = priors,
    ),
    data = (y = data_multivariate,),
    constraints = MeanField(),
    initialization = init,
    iterations = 50,
    free_energy = true
)
record!(r, "multi:m", results_multivariate.posteriors[:m])
record!(r, "multi:w", results_multivariate.posteriors[:w])
record!(r, "multi:s", results_multivariate.posteriors[:s])
record!(r, "multi:z", results_multivariate.posteriors[:z][end])
record!(r, "multi:free_energy", results_multivariate.free_energy)

save_results(r, "gmm", SIDE)
