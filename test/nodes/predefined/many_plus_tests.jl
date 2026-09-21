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

    @testset "Point-mass entropy bookkeeping" begin
        output = NormalMeanVariance(0.7, 0.9)
        gaussian = NormalMeanPrecision(-0.4, 2.0)
        constant = PointMass(2.0)
        for inputs in ((gaussian, constant), (constant, gaussian))
            binary_joint = @call_marginalrule typeof(+)(:in1_in2) (
                m_out = output, m_in1 = inputs[1], m_in2 = inputs[2]
            )
            expected =
                -score(
                    DifferentialEntropy(), Marginal(binary_joint, false, false)
                )
            actual = ReactiveMP._manyplus_negative_entropy(output, inputs)
            @test BayesBase.value(actual) ≈ BayesBase.value(expected)
            @test BayesBase.infinities(actual) ==
                BayesBase.infinities(expected) ==
                1
        end

        # Constants change the joint mean, but only the Gaussian dimensions
        # enter the finite entropy. Each constant contributes one infinity.
        inputs = (
            PointMass(2),
            gaussian,
            PointMass(-3.0f0),
            NormalMeanVariance(0.5, 0.25),
        )
        precision = Diagonal([2.0, 4.0]) + fill(inv(0.9), 2, 2)
        joint = MvNormalWeightedMeanPrecision(zeros(2), Matrix(precision))
        actual = ReactiveMP._manyplus_negative_entropy(output, inputs)
        @test BayesBase.value(actual) ≈ -entropy(joint)
        @test BayesBase.infinities(actual) == 2

        for T in (Float32, Float64, BigFloat)
            constants = (PointMass(T(2)), PointMass(T(-3)), PointMass(4))
            result = ReactiveMP._manyplus_negative_entropy(
                NormalMeanVariance(T(0), T(1)), constants
            )
            @test result isa ReactiveMP.CountingReal{T}
            @test iszero(BayesBase.value(result))
            @test BayesBase.infinities(result) == 3
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

@testitem "nodes:ManyPlus:constant input streams" begin
    using ReactiveMP, BayesBase, ExponentialFamily, Rocket, LinearAlgebra

    variables = (
        randomvar(), constvar(2), randomvar(), constvar(-1.0f0), randomvar()
    )
    node = ReactiveMP.factornode(
        ManyPlus,
        [(:out, variables[1]); [(:inputs, v) for v in Base.tail(variables)]],
        nothing,
    )
    random_variables = variables[[1, 3, 5]]
    sources = map(_ -> Subject(Message), random_variables)
    foreach(zip(random_variables, sources)) do (variable, source)
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
    received = [Any[] for _ in 1:3]
    interfaces = (node.out, node.inputs[2], node.inputs[4])
    subscriptions = map(zip(interfaces, received)) do (interface, values)
        subscribe!(
            ReactiveMP.get_stream_of_outbound_messages(interface),
            message -> push!(values, getdata(ReactiveMP.as_message(message))),
        )
    end
    scores = []
    score_subscription = subscribe!(
        score(
            ReactiveMP.CountingReal{Float64},
            FactorBoundFreeEnergy(),
            node,
            nothing,
            nothing,
        ),
        value -> push!(scores, value),
    )

    for (iteration, scale) in enumerate((1.0, 2.0))
        messages = (
            NormalMeanVariance(5scale, 1.25scale),
            NormalMeanVariance(0.5scale, 0.25scale),
            NormalMeanVariance(-0.25scale, 0.5scale),
        )
        foreach(zip(sources, messages)) do (source, message)
            next!(source, Message(message, false, false))
        end
        @test all(values -> length(values) == iteration, received)
        @test collect(mean_var(last(received[1]))) ≈ [1 + 0.25scale, 0.75scale]
        @test collect(mean_var(last(received[2]))) ≈ [5.25scale - 1, 1.75scale]
        @test collect(mean_var(last(received[3]))) ≈ [4.5scale - 1, 1.5scale]
        @test length(scores) == iteration
        precision =
            Diagonal(inv.(scale .* [0.25, 0.5])) + fill(inv(1.25scale), 2, 2)
        joint = MvNormalWeightedMeanPrecision(zeros(2), Matrix(precision))
        @test BayesBase.value(last(scores)) ≈ -entropy(joint)
        @test BayesBase.infinities(last(scores)) == 2
    end

    foreach(unsubscribe!, subscriptions)
    unsubscribe!(score_subscription)
end
