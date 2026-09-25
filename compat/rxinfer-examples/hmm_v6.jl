# "Basic Examples/Hidden Markov Model": a three-state HMM with DirichletCollection priors on the
# transition and emission matrices, structured VMP q(s_0, s)q(A)q(B), free energy.
# RxInfer 5.5.2 on ReactiveMP 6.5.0.
using RxInfer
include(joinpath(@__DIR__, "common.jl"))

const SIDE = "v6"
r = Recorder()

N = 100
x_data, s_data = hmm_generate_data(N)
record!(r, "data:x", reduce(vcat, x_data))

@model function hidden_markov_model(x)
    A ~ DirichletCollection(ones(3, 3))
    B ~ DirichletCollection(
        [
            10.0 1.0 1.0;
            1.0 10.0 1.0;
            1.0 1.0 10.0
        ]
    )
    s_0 ~ Categorical(fill(1.0 / 3.0, 3))
    s_prev = s_0
    for t in eachindex(x)
        s[t] ~ DiscreteTransition(s_prev, A)
        x[t] ~ DiscreteTransition(s[t], B)
        s_prev = s[t]
    end
end

@constraints function hidden_markov_model_constraints()
    q(s_0, s, A, B) = q(s_0, s)q(A)q(B)
end

imarginals = @initialization begin
    q(A) = vague(DirichletCollection, (3, 3))
    q(B) = vague(DirichletCollection, (3, 3))
    q(s) = vague(Categorical, 3)
end

ireturnvars = (
    A = KeepLast(),
    B = KeepLast(),
    s = KeepLast(),
)

result = infer(
    model = hidden_markov_model(),
    data = (x = x_data,),
    constraints = hidden_markov_model_constraints(),
    initialization = imarginals,
    returnvars = ireturnvars,
    iterations = 20,
    free_energy = true
)
record!(r, "A", result.posteriors[:A])
record!(r, "B", result.posteriors[:B])
record!(r, "s", result.posteriors[:s])
record!(r, "free_energy", result.free_energy)

save_results(r, "hmm", SIDE)
