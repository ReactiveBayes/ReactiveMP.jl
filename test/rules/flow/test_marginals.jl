module RulesFlowMarginalsTest

using Test
using ReactiveMP
using Random
using Distributions
using LinearAlgebra

import ReactiveMP: @test_marginalrules


@testset "marginalrules:Flow" begin

    @testset ":in (m_out::NormalDistributionsFamily, m_in::NormalDistributionsFamily) (Linearization)" begin

        params = [1.0, 2.0, 3.0]
        model = FlowModel( 2, (AdditiveCouplingLayer(PlanarFlow(); permute=false), ) )
        meta  = FlowMeta(compile(model, params))

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

    @testset ":in (m_out::NormalDistributionsFamily, m_in::NormalDistributionsFamily) (Unscented)" begin

        params = [1.0, 2.0, 3.0]
        model = FlowModel( 2, (AdditiveCouplingLayer(PlanarFlow(); permute=false), ) )
        meta  = FlowMeta(compile(model, params), Unscented(2))

        @test_marginalrules [ with_float_conversions = false, atol=1e-9 ] Flow(:in) [
            (
                input = (
                    m_out = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])), 
                    m_in  = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.750029103730993, -2.0000241143779007], [2.5000066521881763 0.5000033260385859; 0.5000033260385859 0.9999999999115444])
            ),
            (
                input = (
                    m_out = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 1/2.0])), 
                    m_in  = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 1/2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.750029103730993, -2.0000241143779007], [2.5000066521881763 0.5000033260385859; 0.5000033260385859 0.9999999999115444])
            ),
            (
                input = (
                    m_out = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 1/2.0])), 
                    m_in  = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 1/2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.750029103730993, -2.0000241143779007], [2.5000066521881763 0.5000033260385859; 0.5000033260385859 0.9999999999115444])
            ),
            (
                input = (
                    m_out = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])), 
                    m_in  = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 1/2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.750029103730993, -2.0000241143779007], [2.5000066521881763 0.5000033260385859; 0.5000033260385859 0.9999999999115444])
            ),
            (
                input = (
                    m_out  = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 1/2.0])), 
                    m_in  = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.750029103730993, -2.0000241143779007], [2.5000066521881763 0.5000033260385859; 0.5000033260385859 0.9999999999115444])
            ),
            (
                input = (
                    m_out = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 1/2.0])), 
                    m_in  = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 1/2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.750029103730993, -2.0000241143779007], [2.5000066521881763 0.5000033260385859; 0.5000033260385859 0.9999999999115444])
            ),
            (
                input = (
                    m_out  = MvNormalMeanPrecision([-5.0, -2.5], diagm([1.0, 1/2.0])), 
                    m_in = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 1/2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.750029103730993, -2.0000241143779007], [2.5000066521881763 0.5000033260385859; 0.5000033260385859 0.9999999999115444])
            ),
            (
                input = (
                    m_out = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])), 
                    m_in = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 1/2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.750029103730993, -2.0000241143779007], [2.5000066521881763 0.5000033260385859; 0.5000033260385859 0.9999999999115444])
            ),
            (
                input = (
                    m_out = MvNormalWeightedMeanPrecision([-5.0, -1.25], diagm([1.0, 1/2.0])), 
                    m_in  = MvNormalMeanCovariance([-5.0, -2.5], diagm([1.0, 2.0])), 
                    meta  = meta
                ), 
                output = MvNormalWeightedMeanPrecision([-10.750029103730993, -2.0000241143779007], [2.5000066521881763 0.5000033260385859; 0.5000033260385859 0.9999999999115444])
            ),

        ]

    end

end



end