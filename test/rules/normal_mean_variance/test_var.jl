module RulesNormalMeanVarianceVarTest

using Test
using ReactiveMP
using Random

import DomainSets
import ReactiveMP: @test_rules

@testset "rules:NormalMeanVariance:var" begin
    @testset "Belief Propagation: (m_out::PointMass, m_μ::UnivariateNormalDistributionsFamily)" begin
        @test_rules [with_float_conversions = true] NormalMeanVariance(:v, Marginalisation) [
            (
                input = (m_out = PointMass(2.0), m_μ = NormalMeanVariance(0.0, 1.0)),
                output = ContinuousUnivariateLogPdf(
                    DomainSets.HalfLine(),
                    (x) -> -log(1.0 + x) / 2 - 4 / (2 * (1.0 + x))
                )
            ),
            (
                input = (m_out = PointMass(0.5), m_μ = NormalMeanVariance(1.0, 0.5)),
                output = ContinuousUnivariateLogPdf(
                    DomainSets.HalfLine(),
                    (x) -> -log(0.5 + x) / 2 - 0.25 / (2 * (0.5 + x))
                )
            ),
            (
                input = (m_out = PointMass(-3.5), m_μ = NormalMeanVariance(0.5, 2.0)),
                output = ContinuousUnivariateLogPdf(
                    DomainSets.HalfLine(),
                    (x) -> -log(2.0 + x) / 2 - 16 / (2 * (2.0 + x))
                )
            ),
            (
                input = (m_out = PointMass(3.5), m_μ = NormalMeanVariance(-0.5, 2.0)),
                output = ContinuousUnivariateLogPdf(
                    DomainSets.HalfLine(),
                    (x) -> -log(2.0 + x) / 2 - 16 / (2 * (2.0 + x))
                )
            )
        ]
    end

    @testset "Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, m_μ::PointMass)" begin
        @test_rules [with_float_conversions = true] NormalMeanVariance(:v, Marginalisation) [
            (
                input = (m_out = NormalMeanVariance(0.0, 1.0), m_μ = PointMass(2.0)),
                output = ContinuousUnivariateLogPdf(
                    DomainSets.HalfLine(),
                    (x) -> -log(1.0 + x) / 2 - 4 / (2 * (1.0 + x))
                )
            ),
            (
                input = (m_out = NormalMeanVariance(1.0, 0.5), m_μ = PointMass(0.5)),
                output = ContinuousUnivariateLogPdf(
                    DomainSets.HalfLine(),
                    (x) -> -log(0.5 + x) / 2 - 0.25 / (2 * (0.5 + x))
                )
            ),
            (
                input = (m_out = NormalMeanVariance(0.5, 2.0), m_μ = PointMass(-3.5)),
                output = ContinuousUnivariateLogPdf(
                    DomainSets.HalfLine(),
                    (x) -> -log(2.0 + x) / 2 - 16 / (2 * (2.0 + x))
                )
            ),
            (
                input = (m_out = NormalMeanVariance(-0.5, 2.0), m_μ = PointMass(3.5)),
                output = ContinuousUnivariateLogPdf(
                    DomainSets.HalfLine(),
                    (x) -> -log(2.0 + x) / 2 - 16 / (2 * (2.0 + x))
                )
            )
        ]
    end

    @testset "Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, m_μ::UnivariateNormalDistributionsFamily)" begin
        @test_rules [with_float_conversions = true] NormalMeanVariance(:v, Marginalisation) [
            (
                input = (m_out = NormalMeanVariance(0.0, 1.0), m_μ = NormalMeanVariance(-2.0, 1.0)),
                output = ContinuousUnivariateLogPdf(
                    DomainSets.HalfLine(),
                    (x) -> -log(2.0 + x) / 2 - 4 / (2 * (2.0 + x))
                )
            ),
            (
                input = (m_out = NormalMeanVariance(1.0, 0.5), m_μ = NormalMeanVariance(0.0, 2.0)),
                output = ContinuousUnivariateLogPdf(
                    DomainSets.HalfLine(),
                    (x) -> -log(2.5 + x) / 2 - 1 / (2 * (2.5 + x))
                )
            ),
            (
                input = (m_out = NormalMeanVariance(0.5, 2.0), m_μ = NormalMeanVariance(1.5, 0.5)),
                output = ContinuousUnivariateLogPdf(
                    DomainSets.HalfLine(),
                    (x) -> -log(2.5 + x) / 2 - 1 / (2 * (2.5 + x))
                )
            ),
            (
                input = (m_out = NormalMeanVariance(-0.5, 2.0), m_μ = NormalMeanVariance(-1.5, 0.5)),
                output = ContinuousUnivariateLogPdf(
                    DomainSets.HalfLine(),
                    (x) -> -log(2.5 + x) / 2 - 1 / (2 * (2.5 + x))
                )
            )
        ]
    end
end

end
