@testitem "Basic out rule tests #1" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    @testset "f(x) = x, x ~ EF" begin
        meta = DeltaMeta(method = CVIProjection(), inverse = nothing)
        m_outs = [
            (NormalMeanVariance(0, 2), NormalMeanVariance(0, 1)),
            (Gamma(2, 2), Gamma(3, 3)),
            (Beta(2, 2), Beta(3, 3)),
            (MvNormalMeanCovariance([0.5, 0.5]), MvNormalMeanCovariance([0.5, 0.5]))
        ]
        for _m_out in m_outs
            m_out_incoming = _m_out[1]
            m_out_outbound = _m_out[2]
            q_out = prod(GenericProd(), m_out_incoming, m_out_outbound)
            q_ins = FactorizedJoint((q_out,)) # `identity`

            msg = @call_rule DeltaFn{identity}(:out, Marginalisation) (m_out = m_out_incoming, q_out = q_out, q_ins = q_ins, meta = meta)

            prj = ProjectedTo(ExponentialFamily.exponential_family_typetag(q_out), size(q_out)...)
            q_out_approximated = prod(GenericProd(), msg, m_out_incoming)
            m_out_projected = project_to(prj, (x) -> logpdf(msg, x))
            q_out_projected = project_to(prj, (x) -> logpdf(q_out_approximated, x))

            @test m_out_projected ≈ m_out_outbound atol = 5e-1
            @test q_out_projected ≈ q_out atol = 5e-1
        end
    end
end

# In this test we are trying to check that `DeltaFn` node can accept arbitrary (univariate) inputs 
# and compute an outbound (multivariate) message. 
# We use a simple node function f(x) = [x, y] and we test the following assumptions:
# - `mean(m_out) ≈ [ mean(m_x), mean(m_y) ]`
@testitem "Basic out rule tests #2" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase
    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)
    @test !isnothing(ext)
    using .ext

    meta = DeltaMeta(method = CVIProjection(), inverse = nothing)
    f(x, y) = [x, y]

    inputs = [
        (NormalMeanVariance(-2.0, 1.0), NormalMeanVariance(3.0, 1.0), MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0])),
        (NormalMeanVariance(3.0, 1.0), NormalMeanVariance(-2.0, 1.0), MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0])),
        (Beta(10.0, 1.0), Beta(1.0, 10.0), MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0]))
    ]

    for input in inputs
        m_x = input[1]
        m_y = input[2]
        m_out_incoming = input[3]
        q_out = input[3]

        # The outbound message rule first requires to compute the joint marginal over inputs
        q_ins = @call_marginalrule DeltaFn{f}(:ins) (m_out = m_out_incoming, m_ins = ManyOf(m_x, m_y), meta = meta)
        m_out_outbound_approximated = @call_rule DeltaFn{f}(:out, Marginalisation) (m_out = m_out_incoming, q_out = q_out, q_ins = q_ins, meta = meta)

        prj = ProjectedTo(ExponentialFamily.exponential_family_typetag(m_out_incoming), size(m_out_incoming)...)
        m_out = project_to(prj, (x) -> logpdf(m_out_outbound_approximated, x))

        @test mean(m_out) ≈ [mean(m_x), mean(m_y)] atol = 2e-1
    end
end