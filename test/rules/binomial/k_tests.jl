@testitem "rules:Binomial:k" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions, DomainSets, SpecialFunctions
    import ForwardDiff: derivative
    import ReactiveMP: @test_rules

    @testset "Variational Message Passing: (q_n::PointMass, q_p::PointMass)" begin
        @test_rules [check_type_promotion = false] Binomial(:k, Marginalisation) [
            (input = (q_n = PointMass(10), q_p = PointMass(0.5)), output = Binomial(10, 0.5)), (input = (q_n = PointMass(3), q_p = PointMass(0.2)), output = Binomial(3, 0.2))
        ]
    end

    @testset "Variational Message Passing: (q_n::PointMass, q_p::Beta)" begin
        @test_rules [check_type_promotion = false] Binomial(:k, Marginalisation) [
            (input = (q_n = PointMass(10), q_p = PointMass(0.5)), output = Binomial(10, 0.5)), (input = (q_n = PointMass(3), q_p = PointMass(0.2)), output = Binomial(3, 0.2))
        ]
    end

    @testset "Variational Message Passing: (q_n, q_p)" begin
        inputs = [(Binomial(20, 0.1), PointMass(0.001)), (Binomial(20, 0.8), Beta(1.0, 0.3)), (Binomial(10, 0.8), Beta(0.8, 3.3)), (Binomial(5, 0.01), Beta(20.5, 18.3))]
        for input in inputs
            η = (mean(log, last(input)) - mean(mirrorlog, last(input)),)
            binom_approximate = convert(Binomial, ExponentialFamilyDistribution(Binomial, η, maximum(support(first(input)))))
            output = @call_rule Binomial(:k, Marginalisation) (q_n = first(input), q_p = last(input))
            @test sum(pdf(output, getsupport(output))) ≈ 1.0
            @test getsupport(output) == support(first(input))

            mean_output = derivative(x -> getlogpartition(output)(x), η[1])
            ## We dont expect the result to be Binomial so mean should not be approximately the Binomial approximation
            @test mean_output ≉ mean(binom_approximate)
        end
    end

    @testset "Sum Product Message Passing: (m_n, m_p)" begin
        Ns = [20, 10, 5]

        for N in Ns
            inputs = [
                (PointMass(N), PointMass(0.3)),
                (PointMass(N), PointMass(0.99)),
                (Binomial(N, 0.1), PointMass(0.1)),
                (Binomial(N, 0.8), Beta(2, 2.3)),
                (Binomial(N, 0.01), PointMass(0.01)),
                (PointMass(N), Beta(3.1, 5.2))
            ]
            for input in inputs
                vmp_output = @call_rule Binomial(:k, Marginalisation) (q_n = first(input), q_p = last(input))
                output = @call_rule Binomial(:k, Marginalisation) (m_n = first(input), m_p = last(input))
                supp_out = Distributions.support(output)
                @test supp_out == DomainSets.Interval(0, N)
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
