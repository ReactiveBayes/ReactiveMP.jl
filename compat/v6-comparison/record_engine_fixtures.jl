# Phase 4.5, Step 0: engine fixtures recorded from full v6 runs, for the new engine to be
# compared against once v6 is gone. One TOML file per slice model under `fixtures/engine/`.
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/record_engine_fixtures.jl
#   julia ... record_engine_fixtures.jl --check     # re-record and compare with the committed files
#   julia ... record_engine_fixtures.jl <id>...     # record, or check, only the models named

using RxInfer, ReactiveMP, Test
using ReactiveMP: getannotations, has_annotation, get_annotation
using MessagePassingRulesTestUtils

const FIXTURES = joinpath(@__DIR__, "fixtures", "engine")
const PACKAGES = Dict("ReactiveMP" => pkgversion(ReactiveMP), "RxInfer" => pkgversion(RxInfer))

node_text(mapping) = string(nameof(typeof(mapping).parameters[1]))

target_text(::Val{S}) where {S} = ":$S"
target_text((tag, index)::Tuple{Val{S}, Int}) where {S} = "(:$S, $index)"

logscale_of(ann) = has_annotation(ann, :logscale) ? Float64(get_annotation(ann, :logscale)) : nothing

# Called with every after-rule-call event when set; `slice_rule_inventory.jl` uses it to
# find which v6 rule each call selected. Recording itself leaves it unset.
const RULE_CALL_HOOK = Ref{Any}(nothing)

# Records every rule call, in the order v6 makes them, tagged with the iteration it
# happened in. A deferred message is computed when it is materialised, so this is
# materialisation order.
function tracing_callbacks()
    trace = RuleCallRecord[]
    iteration = Ref(0)
    callbacks = (
        before_iteration = (event) -> (iteration[] += 1; nothing),
        after_message_rule_call = (event) -> begin
            push!(trace, RuleCallRecord(iteration[], node_text(event.mapping), target_text(event.mapping.vtag), event.result, logscale_of(event.annotations)))
            RULE_CALL_HOOK[] === nothing || RULE_CALL_HOOK[](event)
            nothing
        end,
    )
    return trace, callbacks
end

unwrap(q::Marginal) = ReactiveMP.getdata(q)
unwrap(q) = q
final(posterior::AbstractVector{<:AbstractVector}) = map(unwrap, last(posterior))
final(posterior::AbstractVector) = unwrap(last(posterior))

function record(id; description, model, data, iterations, returnvars, annotations = nothing, free_energy = true, kwargs...)
    trace, callbacks = tracing_callbacks()
    result = infer(; model, data, iterations, free_energy, callbacks, annotations, returnvars = KeepEach(), kwargs...)
    posteriors = Dict{String, Any}(string(name) => final(result.posteriors[name]) for name in returnvars)
    if annotations !== nothing
        for name in returnvars
            q = last(result.posteriors[name])
            q isa Marginal && (posteriors["logscale($name)"] = logscale_of(getannotations(q)))
        end
    end
    return EngineTrajectory(id; description, free_energy = free_energy ? Float64.(result.free_energy) : Float64[], posteriors, trace)
end

@model function bp_iid(y, prior_v, v)
    x ~ NormalMeanVariance(0.0, prior_v)
    for i in eachindex(y)
        y[i] ~ NormalMeanVariance(x, v)
    end
end

@model function bp_chain(y, prior_v, v)
    x[1] ~ NormalMeanVariance(0.0, prior_v)
    y[1] ~ NormalMeanVariance(x[1], v)
    for i in 2:length(y)
        x[i] ~ NormalMeanVariance(x[i - 1], v)
        y[i] ~ NormalMeanVariance(x[i], v)
    end
end

@model function vmp_meanfield(y)
    μ ~ NormalMeanPrecision(0.0, 0.01)
    τ ~ GammaShapeRate(1.0, 1.0)
    for i in eachindex(y)
        y[i] ~ NormalMeanPrecision(μ, τ)
    end
end

@model function vmp_structured(y)
    μ ~ NormalMeanPrecision(0.0, 0.01)
    τ ~ GammaShapeRate(1.0, 1.0)
    for i in eachindex(y)
        x[i] ~ NormalMeanPrecision(μ, τ)
        y[i] ~ NormalMeanVariance(x[i], 0.5)
    end
end

@model function normal_mixture(y)
    π ~ Dirichlet([1.0, 1.0])
    m[1] ~ NormalMeanPrecision(-1.0, 0.1)
    m[2] ~ NormalMeanPrecision(1.0, 0.1)
    p[1] ~ GammaShapeRate(1.0, 1.0)
    p[2] ~ GammaShapeRate(1.0, 1.0)
    for i in eachindex(y)
        z[i] ~ Categorical(π)
        y[i] ~ NormalMixture(switch = z[i], m = m, p = p)
    end
end

@model function mixture_bp(y)
    s ~ Categorical([0.3, 0.7])
    x[1] ~ NormalMeanVariance(-2.0, 1.0)
    x[2] ~ NormalMeanVariance(2.0, 1.0)
    z ~ Mixture(switch = s, inputs = x)
    y ~ NormalMeanVariance(z, 0.5)
end

square_plus_one(x) = x^2 + 1.0

@model function delta_unscented(y)
    x ~ NormalMeanVariance(0.5, 1.0)
    z := square_plus_one(x)
    y ~ NormalMeanVariance(z, 0.1)
end

# Static inputs: the constant 2.0 and the data `s` are folded into the node function, and
# every update waits for them.
scaled_square_plus(c, x, s) = c * x^2 + s

@model function delta_unscented_static(y, s)
    x ~ NormalMeanVariance(0.5, 1.0)
    z := scaled_square_plus(2.0, x, s)
    y ~ NormalMeanVariance(z, 0.1)
end

# Linearization of a cubic. One random input: with two, v6 computes the first input's prior
# message once for each of its subscribers, the same value three times, which a call-by-call
# comparison cannot match.
cube_minus(x) = x^3 - x

@model function delta_linearization(y)
    x ~ NormalMeanVariance(0.5, 1.0)
    z := cube_minus(x)
    y ~ NormalMeanVariance(z, 0.1)
end

# GCV under mean-field: the variance of y about x is exp(z - 0.5), with z unknown, and y itself
# observed through a narrow normal, so that q(y) is a normal and v6's energy applies.
@model function gcv_meanfield(o)
    x ~ NormalMeanVariance(0.5, 1.0)
    z ~ NormalMeanVariance(0.0, 1.0)
    y ~ GCV(x, z, 1.0, -0.5)
    o ~ NormalMeanVariance(y, 0.1)
end

# v6's ExponentialLinearQuadratic has no `params`, which the fixture encoder reads; the port's
# defines them as its four coefficients, and so does this, for v6's type only.
Distributions.params(d::ReactiveMP.ExponentialLinearQuadratic) = (d.a, d.b, d.c, d.d)

# Two Probit outputs of one weight: each rule towards `w` reads the message on its own edge,
# which the other's message feeds, so they start from Probit's default initial message.
@model function probit_ep(y)
    w ~ NormalMeanVariance(0.0, 1.0)
    y[1] ~ Probit(w)
    y[2] ~ Probit(w)
end

# GaussianCoupling's improper message towards `c`, made proper by the observation's likelihood.
@model function gaussian_coupling(y)
    x ~ NormalMeanPrecision(0.5, 2.0)
    c ~ GaussianCoupling(x, 0.5)
    y ~ NormalMeanVariance(c, 1.0)
end

# Belief propagation through deterministic nodes under the default scheme (Phase 5, step 4): a
# tree of the four logic nodes, closed by a Bernoulli factor on its last output.
@model function logic_bp(p)
    x ~ Bernoulli(p)
    y ~ Bernoulli(0.6)
    z ~ AND(x, y)
    n ~ NOT(z)
    w ~ Bernoulli(0.4)
    o ~ OR(n, w)
    v ~ Bernoulli(0.5)
    i ~ IMPLY(o, v)
    i ~ Bernoulli(0.9)
end

const Y = [1.2, 0.7, 2.1, 1.6, 0.9, 1.4]
const Y_MIXTURE = [-2.1, -1.8, 2.2, 1.9, -2.3, 2.0, 1.7, -1.9]

const MODELS = [
    (
        "bp_iid",
        """x ~ NMV(0, 10), y[i] ~ NMV(x, 1) observed; BP, with v6's LogScaleAnnotations. The only
        slice model whose every rule sets a log scale in v6, so the only one recorded with them.""",
        () -> record("bp_iid"; description = "", model = bp_iid(), data = (y = Y, prior_v = 10.0, v = 1.0), iterations = 2, returnvars = (:x,), annotations = (LogScaleAnnotations(),)),
    ),
    (
        "bp_iid_missing",
        "bp_iid with the second observation missing: its rule call is skipped and yields `missing`. No annotations: predicting the missing y[2] calls NormalMeanVariance(:out) with (q_μ::Normal, q_v::PointMass), which sets no log scale in v6. RxInfer treats a missing observation as a prediction, which it will not combine with free energy, so none is recorded.",
        () -> record("bp_iid_missing"; description = "", model = bp_iid(), data = (y = [Y[1], missing, Y[3:end]...], prior_v = 10.0, v = 1.0), iterations = 2, returnvars = (:x,), free_energy = false),
    ),
    (
        "bp_chain",
        "x[1] ~ NMV(0, 10), x[i] ~ NMV(x[i-1], 1), y[i] ~ NMV(x[i], 1) observed; BP along a chain, no annotations.",
        () -> record("bp_chain"; description = "", model = bp_chain(), data = (y = Y, prior_v = 10.0, v = 1.0), iterations = 2, returnvars = (:x,)),
    ),
    (
        "vmp_meanfield",
        "μ ~ NMP(0, 0.01), τ ~ GammaShapeRate(1, 1), y[i] ~ NMP(μ, τ) observed; q(μ)q(τ).",
        () -> record("vmp_meanfield"; description = "", model = vmp_meanfield(), data = (y = Y,), iterations = 5, returnvars = (:μ, :τ), constraints = MeanField(), initialization = @initialization(q(τ) = GammaShapeRate(1.0, 1.0))),
    ),
    (
        "vmp_structured",
        "μ ~ NMP(0, 0.01), τ ~ GammaShapeRate(1, 1), x[i] ~ NMP(μ, τ), y[i] ~ NMV(x[i], 0.5) observed; q(x, μ)q(τ).",
        () -> record(
            "vmp_structured"; description = "", model = vmp_structured(), data = (y = Y,), iterations = 5, returnvars = (:μ, :τ, :x),
            constraints = @constraints(
                begin
                    q(x, μ, τ) = q(x, μ)q(τ)
                end
            ),
            initialization = @initialization(
                begin
                    q(τ) = GammaShapeRate(1.0, 1.0)
                    q(μ) = NormalMeanPrecision(0.0, 1.0)
                end
            ),
        ),
    ),
    (
        "normal_mixture",
        "π ~ Dirichlet([1, 1]), m[k] ~ NMP(∓1, 0.1), p[k] ~ GammaShapeRate(1, 1), z[i] ~ Categorical(π), y[i] ~ NormalMixture(z[i], m, p) observed; mean-field.",
        () -> record(
            "normal_mixture"; description = "", model = normal_mixture(), data = (y = Y_MIXTURE,), iterations = 5, returnvars = (:π, :m, :p, :z), constraints = MeanField(),
            initialization = @initialization(
                begin
                    q(π) = Dirichlet([1.0, 1.0])
                    q(m) = [NormalMeanPrecision(-1.0, 0.1), NormalMeanPrecision(1.0, 0.1)]
                    q(p) = [GammaShapeRate(1.0, 1.0), GammaShapeRate(1.0, 1.0)]
                end
            ),
        ),
    ),
    (
        "delta_unscented",
        "x ~ NMV(0.5, 1), z := x^2 + 1 (Unscented), y ~ NMV(z, 0.1) observed at 2.0.",
        () -> record(
            "delta_unscented"; description = "", model = delta_unscented(), data = (y = 2.0,), iterations = 3, returnvars = (:x, :z),
            meta = @meta(
                begin
                    square_plus_one() -> DeltaMeta(method = Unscented())
                end
            ),
            initialization = @initialization(q(z) = NormalMeanVariance(1.0, 1.0)),
        ),
    ),
    (
        "delta_linearization",
        "x ~ NMV(0.5, 1), z := x^3 - x (Linearization), y ~ NMV(z, 0.1) observed at 2.0.",
        () -> record(
            "delta_linearization"; description = "", model = delta_linearization(), data = (y = 2.0,), iterations = 3, returnvars = (:x, :z),
            meta = @meta(
                begin
                    cube_minus() -> DeltaMeta(method = Linearization())
                end
            ),
            initialization = @initialization(q(z) = NormalMeanVariance(1.0, 1.0)),
        ),
    ),
    (
        "gcv_meanfield",
        "x ~ NMV(0.5, 1), z ~ NMV(0, 1), y ~ GCV(x, z, κ = 1, ω = -0.5), o ~ NMV(y, 0.1) observed at 2; mean-field, the message towards z an ExponentialLinearQuadratic.",
        () -> record(
            "gcv_meanfield"; description = "", model = gcv_meanfield(), data = (o = 2.0,), iterations = 5, returnvars = (:x, :y, :z), constraints = MeanField(),
            initialization = @initialization(
                begin
                    q(x) = NormalMeanVariance(0.5, 1.0)
                    q(y) = NormalMeanVariance(2.0, 1.0)
                    q(z) = NormalMeanVariance(0.0, 1.0)
                end
            ),
        ),
    ),
    (
        "probit_ep",
        "w ~ NMV(0, 1), y[1] ~ Probit(w) and y[2] ~ Probit(w) observed at 1 and 0; expectation propagation, each rule towards w starting from Probit's default initial message NMP(0, 100).",
        () -> record("probit_ep"; description = "", model = probit_ep(), data = (y = [1.0, 0.0],), iterations = 5, returnvars = (:w,)),
    ),
    (
        "gaussian_coupling",
        "x ~ NMP(0.5, 2), c ~ GaussianCoupling(x, 0.5), y ~ NMV(c, 1) observed at 1.5; the message towards c is improper, the likelihood's precision makes c's marginal proper.",
        () -> record("gaussian_coupling"; description = "", model = gaussian_coupling(), data = (y = 1.5,), iterations = 3, returnvars = (:x, :c)),
    ),
    (
        "logic_bp",
        "x ~ Bernoulli(p = 0.3) as data, y ~ Bernoulli(0.6), z ~ AND(x, y), n ~ NOT(z), w ~ Bernoulli(0.4), o ~ OR(n, w), v ~ Bernoulli(0.5), i ~ IMPLY(o, v), and i ~ Bernoulli(0.9); BP through the logic nodes.",
        () -> record("logic_bp"; description = "", model = logic_bp(), data = (p = 0.3,), iterations = 2, returnvars = (:x, :y, :z, :n, :w, :o, :v, :i)),
    ),
    (
        "delta_unscented_static",
        "x ~ NMV(0.5, 1), z := 2x^2 + s (Unscented), with the constant 2 and the data s = 1 folded into the node function, y ~ NMV(z, 0.1) observed at 3.0.",
        () -> record(
            "delta_unscented_static"; description = "", model = delta_unscented_static(), data = (y = 3.0, s = 1.0), iterations = 3, returnvars = (:x, :z),
            meta = @meta(
                begin
                    scaled_square_plus() -> DeltaMeta(method = Unscented())
                end
            ),
            initialization = @initialization(q(z) = NormalMeanVariance(1.0, 1.0)),
        ),
    ),
    (
        "mixture_bp",
        "s ~ Categorical([0.3, 0.7]), x[1] ~ NMV(-2, 1), x[2] ~ NMV(2, 1), z ~ Mixture(switch = s, inputs = x), y ~ NMV(z, 0.5) observed at 1.5; BP with v6's LogScaleAnnotations, which the Mixture rules need (a data variable's message carries no log scale in v6, hence y through NMV). No free energy: v6's Mixture energy is a placeholder returning 0.0, and the port defines none.",
        () -> record("mixture_bp"; description = "", model = mixture_bp(), data = (y = 1.5,), iterations = 2, returnvars = (:s, :x), annotations = (LogScaleAnnotations(),), free_energy = false),
    ),
]

# Found while recording, and deliberately preserved rather than fixed — log scales are a
# niche feature, to be revisited as its own milestone after the migration.
const NOTES = """
Recorded from ReactiveMP $(PACKAGES["ReactiveMP"]) through RxInfer $(PACKAGES["RxInfer"]), unmodified.
Log scales are recorded only for bp_iid and mixture_bp. v6 cannot annotate the others: NormalMeanVariance(:μ)
with (m_out::Normal, q_v::PointMass) sets no @logscale (mean.jl:30) while its mirror out.jl:40
does. Found by reading, and not triggered by these models: LogScaleAnnotations' all-point-mass
fallback (logscale.jl:45-49) covers all-messages or all-marginals, never a mix of the two."""

describe(id) = only(d for (i, d, _) in MODELS if i == id)

function run_all(; check::Bool, only = String[])
    mkpath(FIXTURES)
    return @testset "v6 engine fixtures" begin
        for (id, description, run) in MODELS
            isempty(only) || id in only || continue
            recorded = run()
            trajectory = EngineTrajectory(id; description = replace(description, "\n" => " "), free_energy = recorded.free_energy, posteriors = recorded.posteriors, trace = recorded.trace)
            path = joinpath(FIXTURES, "$id.toml")
            if check
                _, committed = load_engine_fixture(path)
                @test compare_engine_trajectory(trajectory, committed; atol = 1.0e-9) === :agree
            else
                save_engine_fixture(path, trajectory; packages = PACKAGES, notes = NOTES)
                println("recorded $id: $(length(trajectory.trace)) rule calls, free energy $(trajectory.free_energy)")
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all(; check = "--check" in ARGS, only = filter(!startswith("--"), ARGS))
end
