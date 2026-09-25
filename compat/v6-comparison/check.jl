# The migration checker end to end: v7 ports compared with their v6 originals in one
# process, fixtures written and read back, and v6 rules verified against their own node
# definitions:
#
#   julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/check.jl

include(joinpath(@__DIR__, "V6Oracle.jl"))

using .V6Oracle, Test
using ExponentialFamily, BayesBase, Distributions
import ReactiveMP
using MessagePassingRulesTestUtils

module V7Port
    using MessagePassingRulesBase, ExponentialFamily, BayesBase

    @define_factor_node(node = NormalMeanVariance, type = Stochastic, interfaces = [:out, :μ, :v])

    @define_message_update_rule(
        node = NormalMeanVariance, target = :out, args = (m[:μ]::PointMass, m[:v]::PointMass),
        body = (args) -> NormalMeanVariance(mean(args.m[:μ]), mean(args.m[:v])),
    )
    @define_message_update_rule(
        node = NormalMeanVariance, target = :out, args = (m[:μ]::UnivariateNormalDistributionsFamily, m[:v]::PointMass),
        logscale = 0,
        body = (args) -> begin
            μ_mean, μ_var = mean_var(args.m[:μ])
            NormalMeanVariance(μ_mean, μ_var + mean(args.m[:v]))
        end,
    )
    @define_message_update_rule(
        node = NormalMeanVariance, target = :μ, args = (m[:out]::UnivariateNormalDistributionsFamily, m[:v]::PointMass),
        logscale = 0,
        body = (args) -> begin
            out_mean, out_var = mean_var(args.m[:out])
            NormalMeanVariance(out_mean, out_var + mean(args.m[:v]))
        end,
    )
end

using MessagePassingRulesBase: RuleArgs, Target, DefaultAlgorithm, getresult, getlogscale, isdefined_logscale
using MessagePassingRulesBase: message_passing_rule

function v7_message_update(node, edge, m, q)
    result = message_passing_rule(node, Target(edge), DefaultAlgorithm(), RuleArgs(m = m, q = q))
    logscale = getlogscale(result)
    return getresult(result), isdefined_logscale(logscale) ? logscale : nothing
end

const CASES = [
    ("NMV:out:pointmass", :out, (μ = PointMass(1.0), v = PointMass(2.0))),
    ("NMV:out:normal", :out, (μ = NormalMeanVariance(0.5, 1.5), v = PointMass(2.0))),
    ("NMV:out:normal-precision-input", :out, (μ = NormalMeanPrecision(0.5, 4.0), v = PointMass(0.25))),
    ("NMV:μ:normal", :μ, (out = NormalMeanVariance(-1.0, 0.5), v = PointMass(3.0))),
]

records = MigrationRecord[]
@testset "v6 comparison" begin
    @testset "ported rules agree with v6" begin
        for (id, edge, m) in CASES
            v7, v7_logscale = v7_message_update(NormalMeanVariance, edge, m, NamedTuple())
            v6, v6_logscale = v6_message_update(NormalMeanVariance, edge, m, NamedTuple())
            push!(records, compare_with_reference(id, v7, v6; inputs = (m = m,), node = "NormalMeanVariance", target = ":$edge", actual_logscale = v7_logscale, reference_logscale = v6_logscale))
        end
        @test all(r -> r.outcome === :agree, records)
    end

    @testset "fixtures round-trip" begin
        path = joinpath(mktempdir(), "normal_mean_variance.jls")
        save_migration_fixtures(path, records; packages = Dict("ReactiveMP" => pkgversion(ReactiveMP)))
        loaded = load_migration_fixtures(path)
        @test map(r -> r.reference, loaded.records) == map(r -> r.reference, records)
        @test loaded.header.packages["ReactiveMP"] == v"6.5.0"
    end
end

# Verification of v6 rules against their own node definitions. A failure here is a finding
# about v6, not about the port: it is printed and collected, and the summary lists it.
const V6_VERIFICATIONS = [
    ("NormalMeanVariance(:out), BP, point masses", NormalMeanVariance, :out, (μ = PointMass(1.0), v = PointMass(2.0)), NamedTuple()),
    ("NormalMeanVariance(:out), BP, normal μ", NormalMeanVariance, :out, (μ = NormalMeanVariance(0.5, 1.5), v = PointMass(2.0)), NamedTuple()),
    ("NormalMeanVariance(:μ), BP, normal out", NormalMeanVariance, :μ, (out = NormalMeanVariance(-1.0, 0.5), v = PointMass(3.0)), NamedTuple()),
    ("NormalMeanVariance(:μ), VMP, point-mass q_v", NormalMeanVariance, :μ, NamedTuple(), (out = NormalMeanVariance(-1.0, 0.5), v = PointMass(3.0))),
    ("NormalMeanVariance(:μ), VMP, InverseGamma q_v", NormalMeanVariance, :μ, NamedTuple(), (out = NormalMeanVariance(-1.0, 0.5), v = InverseGamma(3.0, 4.0))),
    ("NormalMeanPrecision(:μ), BP, normal out", NormalMeanPrecision, :μ, (out = NormalMeanVariance(-1.0, 0.5), τ = PointMass(2.0)), NamedTuple()),
    ("NormalMeanPrecision(:μ), VMP, Gamma q_τ", NormalMeanPrecision, :μ, NamedTuple(), (out = NormalMeanVariance(-1.0, 0.5), τ = GammaShapeRate(3.0, 2.0))),
    ("Gamma(:out), BP, point masses", Gamma, :out, (α = PointMass(2.0), θ = PointMass(1.5)), NamedTuple()),
    ("Beta(:out), BP, point masses", Beta, :out, (a = PointMass(2.0), b = PointMass(3.0)), NamedTuple()),
    ("Bernoulli(:p), BP, point-mass out", Bernoulli, :p, (out = PointMass(1.0),), NamedTuple()),
    ("Bernoulli(:p), VMP, Bernoulli q_out", Bernoulli, :p, NamedTuple(), (out = Bernoulli(0.3),)),
]

mutable struct CollectingTestSet <: Test.AbstractTestSet
    description::String
    results::Vector{Any}
end
CollectingTestSet(description; kwargs...) = CollectingTestSet(description, Any[])
Test.record(set::CollectingTestSet, result) = (push!(set.results, result); result)
Test.finish(set::CollectingTestSet) = set

findings = Dict{String, Vector{String}}()
for (label, fform, edge, m, q) in V6_VERIFICATIONS
    set = @testset CollectingTestSet "v6: $label" begin
        verify_message_update((m, q) -> v6_message_update(fform, edge, m, q), v6_logdensity(fform), v6_interfaces(fform), edge; m, q, source = LineNumberNode(@__LINE__, Symbol(@__FILE__)))
    end
    failed = [sprint(show, r) for r in set.results if r isa Test.Fail || r isa Test.Error]
    passed = count(r -> r isa Test.Pass, set.results)
    println(isempty(failed) ? "  ✓ " : "  ✗ ", label, " (", passed, " check", passed == 1 ? "" : "s", " passed)")
    isempty(failed) || (findings[label] = failed)
end

# Findings about v6 itself, each investigated. The list is pinned: a new one fails the run
# until it is understood and added here with its explanation.
const KNOWN_V6_FINDINGS = Dict(
    "NormalMeanVariance(:μ), VMP, InverseGamma q_v" =>
        "v6 returns variance E[v]; naive VMP, exp E_q[log N(out | μ, v)], gives 1/E[1/v]. The two agree " *
        "only when q_v is a point mass. For InverseGamma(3, 4) that is 2 against 4/3. Confirmed below by " *
        "verifying the corrected formula. Reported as ReactiveMP.jl#669; to be ported as a :correction in Phase 5.",
)

@testset "the corrected NormalMeanVariance(:μ) VMP rule verifies" begin
    q = (out = NormalMeanVariance(-1.0, 0.5), v = InverseGamma(3.0, 4.0))
    corrected = (m, q) -> (NormalMeanVariance(mean(q.out), 1 / (shape(q.v) / scale(q.v))), nothing)
    verify_message_update(corrected, v6_logdensity(NormalMeanVariance), v6_interfaces(NormalMeanVariance), :μ; q, source = LineNumberNode(@__LINE__, Symbol(@__FILE__)))
end

@testset "v6 verification findings are known" begin
    for (label, failures) in findings
        known = haskey(KNOWN_V6_FINDINGS, label)
        known || println("\nUNEXPLAINED v6 finding: ", label, "\n", join(failures, "\n"))
        @test known
    end
    @test Set(keys(findings)) == Set(keys(KNOWN_V6_FINDINGS))
end
