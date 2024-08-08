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
                @test supp_out == N:(N + floor(logfactorial(N)))
                @test sum(x -> pdf(output, x), supp_out) ≈ 1.0
            end
        end
    end

    @testset "Sum Product Message Passing: (m_k, m_p)" begin
        Ns = [20, 10, 5]

        for N in Ns
            inputs = [(PointMass(N), PointMass(0.3)), (Binomial(N, 0.1), PointMass(0.8)), (Binomial(N, 0.8), Beta(2, 2.3)), (PointMass(N), Beta(3.1, 5.2))]
            for input in inputs
                vmp_output = @call_rule Binomial(:n, Marginalisation) (q_k = first(input), q_p = last(input))
                output = @call_rule Binomial(:n, Marginalisation) (m_k = first(input), m_p = last(input))
                supp_out = support(output)
                @test supp_out == DomainSets.Interval(N, N + floor(logfactorial(N)))
                normalization = sum(x -> pdf(output, x), leftendpoint(supp_out):rightendpoint(supp_out))
                pdf_normalized = (x) -> pdf(output, x) / normalization

                @test sum(pdf_normalized, leftendpoint(supp_out):rightendpoint(supp_out)) ≈ 1.0
                entropy_sp_message = -sum(map(x -> pdf_normalized(x) * log(pdf_normalized(x)), leftendpoint(supp_out):rightendpoint(supp_out)))
                entropy_vmp_message = -sum(map(x -> logpdf(vmp_output, x) == -Inf ? 0 : pdf(vmp_output, x) * logpdf(vmp_output, x), leftendpoint(supp_out):rightendpoint(supp_out)))
                @test entropy_vmp_message ≤ entropy_sp_message
            end
        end
    end
end
