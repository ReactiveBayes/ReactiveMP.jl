@testitem "rules:Binomial:n" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions, DomainSets
    using SpecialFunctions
    import ReactiveMP: @test_rules

    @testset "Variational Message Passing: (q_k, q_p)" begin
        Ns = [20, 10, 5]
        for N in Ns
            inputs = [
                (PointMass(N), PointMass(0.1)),
                (PointMass(N), PointMass(0.99)),
                (Binomial(N, 0.1), PointMass(0.001)),
                (Binomial(N, 0.8), Beta(0.01, 0.3)),
                (PointMass(N), Beta(0.001, 0.2))
            ]
            for input in inputs
                output = @call_rule Binomial(:n, Marginalisation) (q_k = first(input), q_p = last(input))
                supp_out = getsupport(output)
                mean_output = sum(x -> pdf(output, x) * x, supp_out)
                @test supp_out == N:(N + floor(logfactorial(N)))
                @test sum(x -> pdf(output, x), supp_out) ≈ 1.0
                @test mean_output ≥ N
            end
        end
    end

    @testset "Sum Product Message Passing: (m_k, m_p)" begin
        Ns = [10, 5]
        ps = [0.99, 0.01]
        pbs = [0.01, 0.4]
        as = [2.0, 5.0]
        bs = [3.0, 9.0]
        for N in Ns, p in ps, pb in pbs, a in as, b in bs
            inputs = [(PointMass(N), PointMass(p)), (Binomial(N, pb), PointMass(p)), (Binomial(N, pb), Beta(a, b))]
            for input in inputs
                output = @call_rule Binomial(:n, Marginalisation) (m_k = first(input), m_p = last(input))
                supp_out = support(output)
                normalization = sum(x -> pdf(output, x), leftendpoint(supp_out):rightendpoint(supp_out))
                pdf_normalized = (x) -> pdf(output, x) / normalization
                entropy_sp_message = -sum(map(x -> pdf_normalized(x) * log(pdf_normalized(x)), leftendpoint(supp_out):rightendpoint(supp_out)))
                mean_sp_message = sum(map(x -> pdf_normalized(x) * x, leftendpoint(supp_out):rightendpoint(supp_out)))
                @test typeof(output) <: DiscreteUnivariateLogPdf
                @test supp_out == DomainSets.Interval(N, N + floor(logfactorial(N)))
                @test sum(pdf_normalized, leftendpoint(supp_out):rightendpoint(supp_out)) ≈ 1.0
                @test -Inf < entropy_sp_message < Inf
                @test !isnan(entropy_sp_message)
                @test !isnan(mean_sp_message)
                @test mean_sp_message ≥ N
            end
        end
    end
end
