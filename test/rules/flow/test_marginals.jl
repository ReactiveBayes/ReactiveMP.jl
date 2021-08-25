module RulesFlowMarginalsTest

using Test
using ReactiveMP
using Random
using Distributions
using LinearAlgebra

import ReactiveMP: @test_marginalrules


@testset "marginalrules:Flow" begin

    @testset ":in (m_out::NormalDistributionsFamily, m_in::NormalDistributionsFamily)" begin

        model = FlowModel( (NiceLayer(PlanarMap(1.0, 2.0, 3.0)), ) )
        meta  = FlowMeta(model)

        @test_marginalrules [ with_float_conversions = false ] Flow(:in) [
            (
                input = (
                    m_out = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])), 
                    m_in  = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.75002245135493, -2.0000174620747515], [2.5000066522408155 0.5000033261093448; 0.5000033261093448 1.0])
            ),
            (
                input = (
                    m_out = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 1/2.0])), 
                    m_in  = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 1/2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.75002245135493, -2.0000174620747515], [2.5000066522408155 0.5000033261093448; 0.5000033261093448 1.0])
            ),
            (
                input = (
                    m_out = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 1/2.0])), 
                    m_in  = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 1/2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.75002245135493, -2.0000174620747515], [2.5000066522408155 0.5000033261093448; 0.5000033261093448 1.0])
            ),
            (
                input = (
                    m_out = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])), 
                    m_in  = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 1/2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.75002245135493, -2.0000174620747515], [2.5000066522408155 0.5000033261093448; 0.5000033261093448 1.0])
            ),
            (
                input = (
                    m_out  = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 1/2.0])), 
                    m_in  = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.75002245135493, -2.0000174620747515], [2.5000066522408155 0.5000033261093448; 0.5000033261093448 1.0])
            ),
            (
                input = (
                    m_out = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 1/2.0])), 
                    m_in  = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 1/2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.75002245135493, -2.0000174620747515], [2.5000066522408155 0.5000033261093448; 0.5000033261093448 1.0])
            ),
            (
                input = (
                    m_out  = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 1/2.0])), 
                    m_in = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 1/2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.75002245135493, -2.0000174620747515], [2.5000066522408155 0.5000033261093448; 0.5000033261093448 1.0])
            ),
            (
                input = (
                    m_out = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])), 
                    m_in = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 1/2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.75002245135493, -2.0000174620747515], [2.5000066522408155 0.5000033261093448; 0.5000033261093448 1.0])
            ),
            (
                input = (
                    m_out = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 1/2.0])), 
                    m_in  = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.75002245135493, -2.0000174620747515], [2.5000066522408155 0.5000033261093448; 0.5000033261093448 1.0])
            ),

        ]

    end

end



end