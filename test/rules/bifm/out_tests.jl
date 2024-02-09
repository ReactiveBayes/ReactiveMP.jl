
@testitem "rules:BIFM:out" begin
    using ReactiveMP, Random, ExponentialFamily, BayesBase

    import ReactiveMP: @test_rules

    @testset "Belief Propagation: (m_in::MultivariateNormalDistributionsFamily, m_zprev::TerminalProdArgument{<:MultivariateNormalDistributionsFamily}, m_znext::MultivariateNormalDistributionsFamily, meta::BIFMMeta)" begin
        meta = BIFMMeta(
            [2.0 0; 0 1], # A
            [3.0 0; 0 2], # B
            [4.0 0; 0 3], # C
            [5.0 0; 0 4], # H
            [6.0 0; 0 5], # BHBt
            [1.0, 2],     # ξz
            [7.0 0; 0 6], # Λz
            [2.0, 3],     # ξztilde
            [8.0 0; 0 7], # Λztilde
            [3.0, 4],     # μu
            [9.0 0; 0 8]  # Σu
        )

        @test_rules [check_type_promotion = true, atol = [Float32 => 1e-2, Float64 => 1e-2, BigFloat => 1e-8]] BIFM(:out, Marginalisation) [
            (
                input = (
                    m_in = MvNormalMeanPrecision([1, 2], [2 0; 0 1]),
                    m_zprev = TerminalProdArgument(MvNormalMeanPrecision([1, 2], [1 0; 0 2])),
                    m_znext = MvNormalMeanPrecision([1, 2], [1 0; 0 2]),
                    meta = meta
                ),
                output = TerminalProdArgument(MvNormalMeanCovariance([-280, -126], [107680.0 0.0; 0.0 3829.4999999999986]))
            ),
            (
                input = (
                    m_in = MvNormalMeanPrecision([3, 4], [6 0; 0 1]),
                    m_zprev = TerminalProdArgument(MvNormalMeanPrecision([1, 6], [2 0; 0 2])),
                    m_znext = MvNormalMeanPrecision([8, 2], [2 0; 0 2]),
                    meta = meta
                ),
                output = TerminalProdArgument(MvNormalMeanCovariance([-280, -462], [53887.99999999999 0.0; 0.0 3829.4999999999986]))
            ),
            (
                input = (
                    m_in = MvNormalMeanPrecision([5, 6], [2 0; 0 5]),
                    m_zprev = TerminalProdArgument(MvNormalMeanPrecision([6, 2], [1 0; 0 1])),
                    m_znext = MvNormalMeanPrecision([1, 9], [1 0; 0 1]),
                    meta = meta
                ),
                output = TerminalProdArgument(MvNormalMeanCovariance([-1872, -130.8], [107680.0 0.0; 0.0 7614.0]))
            )
        ]
    end
end
