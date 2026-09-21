@testitem "nodes:ManyPlus:construction and dependencies" begin
    using ReactiveMP

    @test ReactiveMP.as_node_symbol(ManyPlus) === :ManyPlus
    @test ReactiveMP.interfaces(ManyPlus) === Val((:out, :inputs))
    @test ReactiveMP.inputinterfaces(ManyPlus) === Val((:inputs,))
    @test ReactiveMP.is_predefined_node(ManyPlus) isa
        ReactiveMP.PredefinedNodeFunctionalForm
    @test ReactiveMP.sdtype(ManyPlus) === Deterministic()

    for ninputs in (0, 1)
        interfaces = [
            (:out, randomvar());
            [(:inputs, randomvar()) for _ in 1:ninputs]
        ]
        @test_throws ArgumentError ReactiveMP.factornode(
            ManyPlus, interfaces, nothing
        )
    end
    @test_throws ArgumentError ReactiveMP.factornode(
        ManyPlus, [(:inputs, randomvar()), (:inputs, randomvar())], nothing
    )

    for ninputs in (2, 3, 7, 64)
        variables = ntuple(_ -> randomvar(), ninputs + 1)
        node = ReactiveMP.factornode(
            ManyPlus,
            [
                (:out, variables[1]);
                [(:inputs, v) for v in Base.tail(variables)]
            ],
            nothing,
        )
        @test functionalform(node) === ManyPlus
        @test ReactiveMP.sdtype(node) === Deterministic()
        @test length(getinterfaces(node)) == ninputs + 1
        @test map(ReactiveMP.getvariable, getinterfaces(node)) === variables
        @test ReactiveMP.getinboundinterfaces(node) === node.inputs
        @test ReactiveMP.interfaceindices(node, (:out, :inputs)) === (1, 2)
        @test ReactiveMP.interfaceindices(node, :inputs) === (2,)
        @test_throws ErrorException ReactiveMP.interfaceindex(node, :invalid)

        deps = ReactiveMP.collect_functional_dependencies(node, nothing)
        @test ReactiveMP.collect_functional_dependencies(node, deps) === deps
        @test_throws ErrorException ReactiveMP.collect_functional_dependencies(
            node, :invalid
        )
        @test ReactiveMP.functional_dependencies(deps, node, node.out, 1) ===
            ((node.inputs,), ())
        for k in (ninputs == 64 ? (1, 32, 64) : (1:ninputs))
            messages, marginals = ReactiveMP.functional_dependencies(
                deps, node, node.inputs[k], k + 1
            )
            @test messages[1] === node.out
            @test messages[2] ===
                Tuple(node.inputs[j] for j in 1:ninputs if j != k)
            @test isempty(marginals)
        end
        @test_throws ErrorException ReactiveMP.functional_dependencies(
            deps, node, node.out, 0
        )
        @test_throws ErrorException ReactiveMP.functional_dependencies(
            deps, node, node.out, ninputs + 2
        )
    end
end

@testitem "nodes:ManyPlus:message streams" begin
    using ReactiveMP, BayesBase, ExponentialFamily, Rocket

    # External message streams let us drive incoming messages independently. A backward
    # message must be available even when the target has never sent a message.
    for ninputs in (2, 3, 7), target in 1:(ninputs + 1)
        variables = ntuple(_ -> randomvar(), ninputs + 1)
        node = ReactiveMP.factornode(
            ManyPlus,
            [
                (:out, variables[1]);
                [(:inputs, v) for v in Base.tail(variables)]
            ],
            nothing,
        )
        interfaces = getinterfaces(node)
        sources = map(_ -> Subject(Message), interfaces)
        foreach(zip(variables, sources)) do (variable, source)
            incoming, _ = ReactiveMP.create_new_stream_of_inbound_messages!(
                variable
            )
            ReactiveMP.connect!(incoming, source)
            ReactiveMP.activate!(
                variable, ReactiveMP.RandomVariableActivationOptions()
            )
        end
        ReactiveMP.activate!(
            node,
            ReactiveMP.FactorNodeActivationOptions(
                nothing, nothing, nothing, nothing, nothing, nothing
            ),
        )
        received = []
        subscription = subscribe!(
            ReactiveMP.get_stream_of_outbound_messages(interfaces[target]),
            message -> push!(received, getdata(ReactiveMP.as_message(message))),
        )
        distributions = ntuple(
            i -> NormalMeanVariance(Float64(i), i / 2), ninputs + 1
        )
        for i in eachindex(sources)
            i == target && continue
            next!(sources[i], Message(distributions[i], false, false))
        end
        @test length(received) == 1
        other_inputs = [i for i in 2:(ninputs + 1) if i != target]
        expected_mean = target == 1 ? sum(other_inputs) : 1 - sum(other_inputs)
        expected_variance =
            target == 1 ? sum(other_inputs) / 2 : (1 + sum(other_inputs)) / 2
        @test mean(only(received)) ≈ expected_mean
        @test var(only(received)) ≈ expected_variance

        # The excluded message must not trigger a recomputation either.
        next!(
            sources[target],
            Message(NormalMeanVariance(100.0, 100.0), false, false),
        )
        @test length(received) == 1
        for i in eachindex(sources)
            i == target && continue
            next!(
                sources[i],
                Message(NormalMeanVariance(2.0i, Float64(i)), false, false),
            )
        end
        @test length(received) == 2
        @test mean(last(received)) ≈ 2expected_mean
        @test var(last(received)) ≈ 2expected_variance
        unsubscribe!(subscription)
    end
end

@testitem "nodes:ManyPlus:free energy" begin
    using ReactiveMP, BayesBase, ExponentialFamily, LinearAlgebra
    import ReactiveMP: @call_marginalrule

    @testset "factor free energy" begin
        output = NormalMeanVariance(0.7, 0.9)
        input1 = NormalMeanPrecision(-0.4, 2.0)
        input2 = NormalWeightedMeanPrecision(0.3, 1.5)

        binary_joint = @call_marginalrule typeof(+)(:in1_in2) (
            m_out = output, m_in1 = input1, m_in2 = input2
        )
        manyplus_score = ReactiveMP._manyplus_negative_entropy(
            output, (input1, input2)
        )
        @test manyplus_score ≈ -entropy(binary_joint)

        for dimension in (2, 3, 8)
            variances = [0.25 + index / 5 for index in 1:dimension]
            output_variance = 0.7
            inputs = Tuple(
                NormalMeanVariance(index / 3, variances[index]) for
                index in 1:dimension
            )

            analytic = ReactiveMP._manyplus_negative_entropy(
                NormalMeanVariance(-0.2, output_variance), inputs
            )
            precision =
                Diagonal(inv.(variances)) +
                fill(inv(output_variance), dimension, dimension)
            dense_joint = MvNormalWeightedMeanPrecision(
                zeros(dimension), Matrix(precision)
            )
            @test analytic ≈ -entropy(dense_joint) rtol = 1e-12 atol = 1e-12
        end
    end
end

@testitem "nodes:ManyPlus:score stream" begin
    using ReactiveMP, BayesBase, ExponentialFamily, Rocket, LinearAlgebra

    variables = ntuple(_ -> randomvar(), 4)
    node = ReactiveMP.factornode(
        ManyPlus,
        [(:out, variables[1]); [(:inputs, v) for v in Base.tail(variables)]],
        nothing,
    )
    interfaces = getinterfaces(node)
    sources = map(_ -> Subject(Message), interfaces)
    foreach(zip(variables, sources)) do (variable, source)
        incoming, _ = ReactiveMP.create_new_stream_of_inbound_messages!(
            variable
        )
        ReactiveMP.connect!(incoming, source)
        ReactiveMP.activate!(
            variable, ReactiveMP.RandomVariableActivationOptions()
        )
    end
    postprocessor = ReactiveMP.ScheduleOnStreamPostprocessor(PendingScheduler())
    scores = score(
        ReactiveMP.CountingReal{Float64},
        FactorBoundFreeEnergy(),
        node,
        nothing,
        postprocessor,
    )
    received = []
    subscription = subscribe!(scores, value -> push!(received, float(value)))
    for source in sources
        next!(source, Message(NormalMeanVariance(0.0, 100.0), false, true))
    end
    release!(postprocessor)
    @test isempty(received)

    for scale in (1.0, 2.0)
        for (i, source) in enumerate(sources)
            next!(
                source,
                Message(
                    NormalMeanVariance(Float64(i), scale * i), false, false
                ),
            )
        end
        previous_count = length(received)
        release!(postprocessor)
        @test length(received) == previous_count + 1
        precision = Diagonal(inv.(scale .* [2, 3, 4])) + fill(inv(scale), 3, 3)
        joint = MvNormalWeightedMeanPrecision(zeros(3), Matrix(precision))
        @test last(received) ≈ -entropy(joint)
    end
    unsubscribe!(subscription)
end
