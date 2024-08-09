@testitem "rules:Binomial:p" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions, LogExpFunctions

    import ReactiveMP: @test_rules

    @testset "Variational Message Passing: (q_k::PointMass, q_n::PointMass)" begin
        @test_rules [check_type_promotion = false] Binomial(:p, Marginalisation) [
            (input = (q_k = PointMass(5), q_n = PointMass(10)), output = Beta(6, 6)), (input = (q_k = PointMass(3), q_n = PointMass(5)), output = Beta(4, 3))
        ]
    end

    @testset "Variational Message Passing: (q_k::Binomial, q_n::PointMass)" begin
        @test_rules [check_type_promotion = false] Binomial(:p, Marginalisation) [
            (input = (q_k = Binomial(5, 0.1), q_n = PointMass(10)), output = Beta(mean(Binomial(5, 0.1)) + 1, 10 - mean(Binomial(5, 0.1)) + 1)),
            (input = (q_k = Binomial(3, 0.4), q_n = PointMass(5)), output = Beta(mean(Binomial(3, 0.4)) + 1, 5 - mean(Binomial(3, 0.4)) + 1))
        ]
    end

    @testset "Variational Message Passing: (q_k::Binomial, q_n::PointMass)" begin
        @test_rules [check_type_promotion = false] Binomial(:p, Marginalisation) [
            (input = (q_k = Binomial(5, 0.1), q_n = PointMass(10)), output = Beta(mean(Binomial(5, 0.1)) + 1, 10 - mean(Binomial(5, 0.1)) + 1)),
            (input = (q_k = Binomial(3, 0.4), q_n = PointMass(5)), output = Beta(mean(Binomial(3, 0.4)) + 1, 5 - mean(Binomial(3, 0.4)) + 1))
        ]
    end

    @testset "Sum Product Message Passing: (m_k::Binomial, m_n::PointMass)" begin
        @test_rules [check_type_promotion = false] Binomial(:p, Marginalisation) [
            (input = (m_k = PointMass(5), m_n = PointMass(10)), output = Beta(5 + 1, 10 - 5 + 1)),
            (input = (q_k = Binomial(3, 0.4), q_n = PointMass(5)), output = Beta(mean(Binomial(3, 0.4)) + 1, 5 - mean(Binomial(3, 0.4)) + 1))
        ]
    end
    @testset "SumProduct Message Passing: (m_k::Binomia, m_n::Binomial)" begin
        ks = [Binomial(5, 0.1), Binomial(4, 0.9), Binomial(9, 0.1)]
        ns = [Binomial(10, 0.9), Binomial(20, 0.01), Binomial(25, 0.23)]
        dt = 0.0001
        grid = collect(0:dt:1)
        for k in ks, n in ns
            output = @call_rule Binomial(:p, Marginalisation) (m_k = k, m_n = n)
            @test typeof(output) <: ContinuousUnivariateLogPdf

            lp = mean(map(x -> pdf(output, x), grid))
            pf = x -> pdf(output, x) / lp
            @test mean(pf.(grid)) ≈ 1.0
            entropy_sp_message = -mean(map(x -> pf(x) * log(pf(x)), grid))
            @test -Inf < entropy_sp_message < Inf
            @test !isnan(entropy_sp_message)
        end
    end
end
