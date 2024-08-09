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
        Ns = [5, 40]
        ps = [0.99, 0.01]
        pbs = [0.01, 0.4, 0.99]
        as = [2.0, 5.0]
        bs = [3.0, 9.0]
        for N in Ns, p in ps, pb in pbs, a in as, b in bs
            inputs = [(PointMass(N), PointMass(p)), (Binomial(N, pb), PointMass(p)), (Binomial(N, pb), Beta(a, b))]
            for input in inputs
                vmp_output = @call_rule Binomial(:k, Marginalisation) (q_n = first(input), q_p = last(input))
                output = @call_rule Binomial(:k, Marginalisation) (m_n = first(input), m_p = last(input))
                supp_out = Distributions.support(output)
                normalization = sum(map(x -> pdf(output, x), 0:N))
                pdf_normalized = (x) -> pdf(output, x) / normalization
                entropy_sp_message = -sum(map(x -> log(pdf_normalized(x)) == -Inf ? 0 : pdf_normalized(x) * log(pdf_normalized(x)), 0:N))
                mean_sp_message = mean(x -> x * pdf_normalized(x), 0:N)
                @test sum(pdf_normalized, 0:N) ≈ 1.0
                @test -Inf < entropy_sp_message < Inf
                @test !isnan(entropy_sp_message)
                @test !isnan(mean_sp_message)
            end
        end
    end
end
