@testitem "Basic out rule tests #1" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext
    
    @testset "f(x) = x, x ~ EF" begin
        meta = DeltaMeta(method = CVIProjection(out_samples_no = 1000), inverse = nothing)
       
        q_ins_m_out_incomings = [
            (FactorizedJoint((NormalMeanVariance(2, 2),)), NormalMeanVariance(0, 1)),
            (FactorizedJoint((Gamma(3, 4),)), Gamma(2,3)),
            (FactorizedJoint((Beta(5, 4),)), Beta( 5, 5)),
            (FactorizedJoint((Rayleigh(4),)), Rayleigh(5.2)),
            (FactorizedJoint((Geometric(0.3),)), Geometric(0.9)),
            (FactorizedJoint((LogNormal(0.2, 1.0),)), LogNormal(3.0, 2.1)),
            (FactorizedJoint((Exponential(0.3),)), Exponential(4.3)),
        ]
   
        for (q_in, m_out_incoming) in q_ins_m_out_incomings
           
            q_in_component = first(components(q_in))
            q_out = q_in_component
            msg = @call_rule DeltaFn{identity}(:out, Marginalisation) (m_out = m_out_incoming, q_out = q_out, q_ins = q_in, meta = meta)

            prj = ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...)
            
            q_out_projected = project_to(prj, (x) -> logpdf(msg, x) +logpdf(m_out_incoming, x))
            @test mean(q_out_projected) ≈ mean(q_out) atol = 5e-1
            @test var(q_out_projected) ≈ var(q_out) atol = 2.0
            @test mode(q_out_projected) ≈ mode(q_out) atol = 5e-1
            if typeof(q_out) <: Union{Exponential, Beta, Gamma}
                @test mean(log, q_out_projected) ≈ mean(log, q_out) atol = 5e-1
            end
        end
    end
end

# In this test we are trying to check that `DeltaFn` node can accept arbitrary (univariate) inputs 
# and compute an outbound (multivariate) message. 
# We use a simple node function f(x) = [x; y] and we test the following assumptions:
# - `mean(m_out) ≈ [ mean(m_x), mean(m_y) ]`
@testitem "Basic out rule tests #2" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase, LinearAlgebra
    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)
    @test !isnothing(ext)
    using .ext

    meta = DeltaMeta(method = CVIProjection(out_samples_no = 1000), inverse = nothing)
    f(x, y) = [x; y]
    q_ins_m_out_incomings = [
        (FactorizedJoint((NormalMeanVariance(3.0, 1.0),MvNormalMeanCovariance([2.0, 5.2], 3diageye(2)))), MvNormalMeanCovariance([2.0, 3.4, -1.0], Diagonal([0.2, 0.01, 4.0]))),
        (FactorizedJoint((MvNormalMeanCovariance([0.3, 0.9], diageye(2)),MvNormalMeanCovariance([2.0, 5.2], 3diageye(2)))), MvNormalMeanCovariance([2.0, 3.4, -1.0, 5.0], Diagonal([0.2, 2.0, 0.01, 4.0]))),
        (FactorizedJoint((MvNormalMeanCovariance([2.0, 3.0, 0.1, 0.9], diageye(4)),MvNormalMeanCovariance([3.4, 7.6], 0.1diageye(2)))), MvNormalMeanCovariance([2.0, 3.4, -1.0, 5.0, 3.0, -10.0], Diagonal([0.2, 2.0, 0.01, 4.0, 1.0, 0.5]))),
    ]
    for (q_in, m_out_incoming) in q_ins_m_out_incomings
           
        q_in_components = components(q_in)
        mean_in_components = mapreduce(mean, vcat, q_in_components)
        cov_in_components  = Diagonal(mapreduce(var, vcat, q_in_components))
        q_out = MvNormalMeanCovariance(mean_in_components, cov_in_components)
        msg = @call_rule DeltaFn{f}(:out, Marginalisation) (m_out = m_out_incoming, q_out = q_out, q_ins = q_in, meta = meta)

        prj = ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...)
        
        q_out_projected = project_to(prj, (x) -> logpdf(msg, x) +logpdf(m_out_incoming, x))
        @test mean(q_out_projected) ≈ mean(q_out) rtol = 5e-1
        @test var(q_out_projected) ≈ var(q_out) rtol = 1.0
        @test mode(q_out_projected) ≈ mode(q_out) rtol = 5e-1
 
    end


end



@testitem "Basic out rule tests #3" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext
    
    @testset "f(x) = a*x + b, x ~ Normal (Univariate)" begin
        meta = DeltaMeta(method = CVIProjection(out_samples_no = 1000), inverse = nothing)
        q_ins = [
            FactorizedJoint((NormalMeanVariance(0, 2),)),
            FactorizedJoint((NormalMeanVariance(3, 4),)),
            FactorizedJoint((NormalMeanVariance(5, 4),))
        ]

        m_out_incomings = [
            NormalMeanVariance(0, 1),
            NormalMeanVariance(2, 3),
            NormalMeanVariance(4, 0.1)
        ]
        constants_b = [-3, -2, -1, 0, 1, 2, 3]
        constants_a = [1, 1.1,]
        for q_in in q_ins, a in constants_a, b in constants_b,m_out_incoming in m_out_incomings
            f(x) = a*x + b
            q_in_component = first(components(q_in))
            q_out = NormalMeanVariance(a*mean(q_in_component) + b, a^2*var(q_in_component))

            msg = @call_rule DeltaFn{f}(:out, Marginalisation) (m_out = m_out_incoming, q_out = q_out, q_ins = q_in, meta = meta)

            prj = ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...)
            
            q_out_projected = project_to(prj, (x) -> logpdf(msg, x) +logpdf(m_out_incoming, x))
            @test mean(q_out_projected) ≈ mean(q_out) atol = 5e-1
            @test var(q_out_projected) ≈ var(q_out) atol = 7e-1
        end
    end
end


@testitem "Basic out rule tests #4" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext
    
    @testset "f(x) = x + constant, x ~ Normal (Multivariate)" begin
        meta = DeltaMeta(method = CVIProjection(out_samples_no = 1000), inverse = nothing)
        q_ins = [
            FactorizedJoint((MvNormalMeanCovariance(zeros(3), 2*diageye(3)),)),
            FactorizedJoint((MvNormalMeanCovariance([0.3, 0.7, 10.0], 0.1*diageye(3)),))
        ]

        m_out_incomings = [
            MvNormalMeanCovariance(zeros(3), 0.4*diageye(3)),
            MvNormalMeanCovariance(ones(3), 0.9*diageye(3))
        ]
        constants = [[1.0, 2.0, 0.4], [0.2, -9.0, 3.0]]
        for q_in in q_ins, c in constants, m_out_incoming in m_out_incomings
            f(x) = x + c
            q_in_component = first(components(q_in))
            q_out = MvNormalMeanCovariance(mean(q_in_component) + c, cov(q_in_component))

            msg = @call_rule DeltaFn{f}(:out, Marginalisation) (m_out = m_out_incoming, q_out = q_out, q_ins = q_in, meta = meta)
            # prj = ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...)
            
            # q_out_projected = project_to(prj, (x) -> logpdf(msg, x) +logpdf(m_out_incoming, x))
            # @test mean(q_out_projected) ≈ mean(q_out) atol = 5e-1
            # @test v(q_out_projected) ≈ var(q_out) atol = 5e-1
        end
    end
end