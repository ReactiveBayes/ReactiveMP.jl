module RulesNormalMeanVarianceVarTest

using Test
using ReactiveMP
using Random

import DomainSets
import ReactiveMP: @test_rules

@testset "rules:NormalMeanVariance:var" begin
    @testset "Belief Propagation: (m_out::PointMass, m_μ::UnivariateNormalDistributionsFamily)" begin
        @test_rules [check_type_promotion = true] NormalMeanVariance(:v, Marginalisation) [
            (
                input = (m_out = PointMass(2.0), m_μ = NormalMeanVariance(0.0, 1.0)),
                output = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -log(1.0 + x) / 2 - 4 / (2 * (1.0 + x)))
            ),
            (
                input = (m_out = PointMass(0.5), m_μ = NormalMeanVariance(1.0, 0.5)),
                output = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -log(0.5 + x) / 2 - 0.25 / (2 * (0.5 + x)))
            ),
            (
                input = (m_out = PointMass(-3.5), m_μ = NormalMeanVariance(0.5, 2.0)),
                output = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -log(2.0 + x) / 2 - 16 / (2 * (2.0 + x)))
            ),
            (
                input = (m_out = PointMass(3.5), m_μ = NormalMeanVariance(-0.5, 2.0)),
                output = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -log(2.0 + x) / 2 - 16 / (2 * (2.0 + x)))
            )
        ]
    end

    @testset "Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, m_μ::PointMass)" begin
        @test_rules [check_type_promotion = true] NormalMeanVariance(:v, Marginalisation) [
            (
                input = (m_out = NormalMeanVariance(0.0, 1.0), m_μ = PointMass(2.0)),
                output = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -log(1.0 + x) / 2 - 4 / (2 * (1.0 + x)))
            ),
            (
                input = (m_out = NormalMeanVariance(1.0, 0.5), m_μ = PointMass(0.5)),
                output = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -log(0.5 + x) / 2 - 0.25 / (2 * (0.5 + x)))
            ),
            (
                input = (m_out = NormalMeanVariance(0.5, 2.0), m_μ = PointMass(-3.5)),
                output = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -log(2.0 + x) / 2 - 16 / (2 * (2.0 + x)))
            ),
            (
                input = (m_out = NormalMeanVariance(-0.5, 2.0), m_μ = PointMass(3.5)),
                output = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -log(2.0 + x) / 2 - 16 / (2 * (2.0 + x)))
            )
        ]
    end

    @testset "Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, m_μ::UnivariateNormalDistributionsFamily)" begin
        @test_rules [check_type_promotion = true] NormalMeanVariance(:v, Marginalisation) [
            (
                input = (m_out = NormalMeanVariance(0.0, 1.0), m_μ = NormalMeanVariance(-2.0, 1.0)),
                output = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -log(2.0 + x) / 2 - 4 / (2 * (2.0 + x)))
            ),
            (
                input = (m_out = NormalMeanVariance(1.0, 0.5), m_μ = NormalMeanVariance(0.0, 2.0)),
                output = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -log(2.5 + x) / 2 - 1 / (2 * (2.5 + x)))
            ),
            (
                input = (m_out = NormalMeanVariance(0.5, 2.0), m_μ = NormalMeanVariance(1.5, 0.5)),
                output = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -log(2.5 + x) / 2 - 1 / (2 * (2.5 + x)))
            ),
            (
                input = (m_out = NormalMeanVariance(-0.5, 2.0), m_μ = NormalMeanVariance(-1.5, 0.5)),
                output = ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -log(2.5 + x) / 2 - 1 / (2 * (2.5 + x)))
            )
        ]
    end

    # f(m_o, v_o, m_w, v_w) = 0.5*(m_o^2 + v_o - 2*m_o*m_w + m_w^2 + v_w)
    @testset "Variational: (q_out::Any, q_μ::Any)" begin
        @test_rules [check_type_promotion = true] NormalMeanVariance(:v, Marginalisation) [
            (input = (q_out = PointMass(-1.0), q_μ = PointMass(2.0)), output = GammaInverse(-0.5, 4.5; check_args = false)),
            (input = (q_out = PointMass(1.0), q_μ = PointMass(2.0)), output = GammaInverse(-0.5, 0.5; check_args = false)),
            (input = (q_out = PointMass(2.0), q_μ = PointMass(1.0)), output = GammaInverse(-0.5, 0.5; check_args = false))
        ]

        @test_rules [check_type_promotion = true] NormalMeanVariance(:v, Marginalisation) [
            (input = (q_out = NormalMeanVariance(-1.0, 2.0), q_μ = PointMass(2.0)), output = GammaInverse(-0.5, 5.5; check_args = false)),
            (input = (q_out = NormalMeanPrecision(1.0, 4.0), q_μ = PointMass(3.0)), output = GammaInverse(-0.5, 2.125; check_args = false)),
            (input = (q_out = NormalWeightedMeanPrecision(2.0, 4.0), q_μ = PointMass(1.0)), output = GammaInverse(-0.5, 0.25; check_args = false))
        ]

        @test_rules [check_type_promotion = true] NormalMeanVariance(:v, Marginalisation) [
            (input = (q_out = PointMass(2.0), q_μ = NormalMeanVariance(-1.0, 2.0)), output = GammaInverse(-0.5, 5.5; check_args = false)),
            (input = (q_out = PointMass(3.0), q_μ = NormalMeanPrecision(1.0, 4.0)), output = GammaInverse(-0.5, 2.125; check_args = false)),
            (input = (q_out = PointMass(1.0), q_μ = NormalWeightedMeanPrecision(2.0, 4.0)), output = GammaInverse(-0.5, 0.25; check_args = false))
        ]

        @test_rules [check_type_promotion = true] NormalMeanVariance(:v, Marginalisation) [
            (input = (q_out = NormalMeanVariance(2.0, 4.0), q_μ = NormalMeanVariance(-1.0, 2.0)), output = GammaInverse(-0.5, 7.5; check_args = false)),
            (input = (q_out = NormalMeanPrecision(3.0, 3.0), q_μ = NormalMeanPrecision(1.0, 4.0)), output = GammaInverse(-0.5, 55.0 / 24.0; check_args = false)),
            (input = (q_out = NormalWeightedMeanPrecision(1.0, 1.0), q_μ = NormalWeightedMeanPrecision(2.0, 4.0)), output = GammaInverse(-0.5, 0.75; check_args = false))
        ]
    end

    # f(m, V) = 0.5*((m[1] - m[2])^2 + V[1,1] - 2*V[1,2] + V[2,2])
    @testset "Variational: (q_out_μ::Any)" begin
        @test_rules [check_type_promotion = true] NormalMeanVariance(:v, Marginalisation) [
            (input = (q_out_μ = MvNormalMeanCovariance([1.0, 1.0], [1.0 0.0; 0.0 1.0]),), output = GammaInverse(-0.5, 1.0; check_args = false)),
            (input = (q_out_μ = MvNormalMeanCovariance([2.0, 3.0], [2.0 -0.1; -0.1 3.0]),), output = GammaInverse(-0.5, 3.1; check_args = false)),
            (input = (q_out_μ = MvNormalMeanCovariance([4.0, 1.0], [4.0 1.0; 1.0 9.0]),), output = GammaInverse(-0.5, 10.0; check_args = false))
        ]

        @test_rules [check_type_promotion = true] NormalMeanVariance(:v, Marginalisation) [
            (input = (q_out_μ = MvNormalMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.0]),), output = GammaInverse(-0.5, 1.0; check_args = false)),
            (input = (q_out_μ = MvNormalMeanPrecision([2.0, 3.0], [2.0 -0.1; -0.1 3.0]),), output = GammaInverse(-0.5, 1079.0 / 1198.0; check_args = false)),
            (input = (q_out_μ = MvNormalMeanPrecision([4.0, 1.0], [4.0 1.0; 1.0 9.0]),), output = GammaInverse(-0.5, 33.0 / 7.0; check_args = false))
        ]

        @test_rules [check_type_promotion = true] NormalMeanVariance(:v, Marginalisation) [
            (input = (q_out_μ = MvNormalWeightedMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.0]),), output = GammaInverse(-0.5, 1.0; check_args = false)),
            (input = (q_out_μ = MvNormalWeightedMeanPrecision([4.0, 1.0], [4.0 1.0; 1.0 9.0]),), output = GammaInverse(-0.5, 5.0 / 7.0; check_args = false))
        ]
    end
end

end
