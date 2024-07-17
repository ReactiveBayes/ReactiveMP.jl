@testitem "Basic out rule tests" begin
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