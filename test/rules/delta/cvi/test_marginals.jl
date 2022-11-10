module RulesCVIMarginalsTest

using Test
using ReactiveMP
using Random
using Distributions
using Flux
using Zygote
using StableRNGs

import ReactiveMP: @test_marginalrules
import ReactiveMP: FactorizedJoint, getmultipliers

struct ZygoteGrad end

function ReactiveMP.compute_grad(::ZygoteGrad, A::F, vec_params) where {F}
    Zygote.gradient(A, vec_params)[1]
end

function ReactiveMP.compute_hessian(::ZygoteGrad, A::G, ::F, vec_params) where {G, F}
    Zygote.hessian(A, vec_params)
end

add_1 = (x::Real) -> x + 1

function two_into_one(x::Real, y::Real)
    return [x, y]
end

function extract_coordinate(x::Vector)
    return x[1]
end

@testset "marginalrules:CVI" begin
    @testset "id, x~Normal, y~Normal" begin
        seed = 123
        rng = StableRNG(seed)
        optimizer = Descent(0.01)
        test_meta = DeltaMeta(method = CVIApproximation(rng, 1, 500, optimizer))
        @test_marginalrules [with_float_conversions = false, atol = 0.1] DeltaFn{identity}(:ins) [
            (
                input = (m_out = NormalMeanVariance(1, 1), m_ins = ManyOf(NormalMeanVariance()), meta = test_meta),
                output = FactorizedJoint((NormalWeightedMeanPrecision(1.0, 2.0),))
            ),
            (
                input = (m_out = NormalMeanVariance(2, 1), m_ins = ManyOf(NormalMeanVariance()), meta = test_meta),
                output = FactorizedJoint((NormalWeightedMeanPrecision(2.0, 2.0),))
            ),
            (
                input = (m_out = NormalMeanVariance(10, 1), m_ins = ManyOf(NormalMeanVariance()), meta = test_meta),
                output = FactorizedJoint((NormalWeightedMeanPrecision(10.0, 2.0),))
            )
        ]
    end
    @testset "id, x ~ MvNormal, y ~ MvNormal" begin
        seed = 123
        rng = StableRNG(seed)
        optimizer = Descent(0.01)
        test_meta = DeltaMeta(method = CVIApproximation(rng, 1, 1000, optimizer))
        @test_marginalrules [with_float_conversions = false, atol = 0.1] DeltaFn{identity}(:ins) [
            (
                input = (m_out = MvGaussianMeanCovariance(ones(2)), m_ins = ManyOf(MvGaussianMeanCovariance(zeros(2))), meta = test_meta),
                output = FactorizedJoint((MvNormalWeightedMeanPrecision(ones(2), diageye(2) * 2),))
            ),
            (
                input = (m_out = MvGaussianMeanCovariance(ones(2) * 10), m_ins = ManyOf(MvGaussianMeanCovariance(zeros(2))), meta = test_meta),
                output = FactorizedJoint((MvNormalWeightedMeanPrecision(ones(2) * 10, diageye(2) * 2),))
            ),
            (
                input = (m_out = MvGaussianMeanCovariance(ones(2), [2 -1; -1 2]), m_ins = ManyOf(MvGaussianMeanCovariance(zeros(2))), meta = test_meta),
                output = FactorizedJoint(((MvNormalWeightedMeanPrecision(ones(2), [1+2 / 3 1/3; 1/3 1+2 / 3])),))
            )
        ]
    end
    @testset "f(x) = x + k, x~Normal, y~Normal" begin
        seed = 123
        rng = StableRNG(seed)
        optimizer = Descent(0.01)
        test_meta = DeltaMeta(method = CVIApproximation(rng, 1, 500, optimizer))
        @test_marginalrules [with_float_conversions = false, atol = 0.1] DeltaFn{add_1}(:ins) [
            (input = (m_out = NormalMeanVariance(1, 1), m_ins = ManyOf(NormalMeanVariance()), meta = test_meta), output = FactorizedJoint((NormalWeightedMeanPrecision(0, 2.0),))),
            (input = (m_out = NormalMeanVariance(2, 1), m_ins = ManyOf(NormalMeanVariance()), meta = test_meta), output = FactorizedJoint((NormalWeightedMeanPrecision(1, 2.0),))),
            (input = (m_out = NormalMeanVariance(10, 1), m_ins = ManyOf(NormalMeanVariance()), meta = test_meta), output = FactorizedJoint((NormalWeightedMeanPrecision(9, 2.0),)))
        ]
    end

    @testset "f(x, y) -> [x, y], x~Normal, y~Normal, out~MvNormal (marginalization)" begin
        seed = 123
        rng = StableRNG(seed)
        optimizer = Descent(0.01)
        test_meta = DeltaMeta(method = CVIApproximation(rng, 1, 2000, optimizer))
        @test_marginalrules [with_float_conversions = false, atol = 0.1] DeltaFn{two_into_one}(:ins) [(
            input = (m_out = MvGaussianMeanCovariance(ones(2), [2 0; 0 2]), m_ins = ManyOf(NormalMeanVariance(), NormalMeanVariance(1, 2)), meta = test_meta),
            output = FactorizedJoint((NormalWeightedMeanPrecision(1 / 2, 1.5), NormalWeightedMeanPrecision(1.0, 1.0)))
        )]
    end

    @testset "f(x) -> x[1], x~MvNormal out~Normal" begin
        seed = 123
        rng = StableRNG(seed)
        optimizer = Descent(0.001)
        test_meta = DeltaMeta(method = CVIApproximation(rng, 1, 10000, optimizer))

        @test_marginalrules [with_float_conversions = false, atol = 0.1] DeltaFn{extract_coordinate}(:ins) [(
            input = (m_out = NormalMeanVariance(0, 1), m_ins = ManyOf(MvGaussianMeanCovariance(ones(2), [1 0; 0 1])), meta = test_meta),
            output = FactorizedJoint((MvNormalWeightedMeanPrecision(ones(2), [2 0; 0 1]),))
        )]
    end

    @testset "id, x~Gamma out~Gamma" begin
        seed = 123
        rng = StableRNG(seed)
        optimizer = Flux.Descent(0.01)
        test_meta = DeltaMeta(method = CVIApproximation(rng, 1, 1000, optimizer))

        @test_marginalrules [with_float_conversions = false, atol = 0.2] DeltaFn{identity}(:ins) [
            (input = (m_out = GammaShapeRate(1, 1), m_ins = ManyOf(GammaShapeRate(1, 1)), meta = test_meta), output = FactorizedJoint((GammaShapeRate(1, 2),))),
            (input = (m_out = GammaShapeRate(1, 1), m_ins = ManyOf(GammaShapeRate(1, 2)), meta = test_meta), output = FactorizedJoint((GammaShapeRate(1, 3),))),
        ]

        seed = 123
        rng = StableRNG(seed)
        optimizer = Flux.Descent(0.001)
        test_meta = DeltaMeta(method = CVIApproximation(rng, 1, 50000, optimizer, ZygoteGrad()))

        @test_marginalrules [with_float_conversions = false, atol = 0.3] DeltaFn{identity}(:ins) [
            (input = (m_out = GammaShapeRate(2, 1), m_ins = ManyOf(GammaShapeRate(1, 2)), meta = test_meta), output = FactorizedJoint((GammaShapeRate(2, 3),))),
            (input = (m_out = GammaShapeRate(2, 2), m_ins = ManyOf(GammaShapeRate(1, 3)), meta = test_meta), output = FactorizedJoint((GammaShapeRate(2, 5),)))
        ]
    end
end
end
