@testitem "rules:ManyPlus" begin
    using ReactiveMP, BayesBase, ExponentialFamily, Distributions

    function manyplus_rule_output(inputs)
        return @call_rule ManyPlus(:out, Marginalisation) (
            m_inputs = ReactiveMP.ManyOf(Tuple(inputs)),
        )
    end

    function manyplus_rule_input(output, other_inputs, target_index)
        messages = (
            ReactiveMP.Message(output, false, false),
            ReactiveMP.ManyOf(
                map(
                    input -> ReactiveMP.Message(input, false, false),
                    Tuple(other_inputs),
                ),
            ),
        )
        return ReactiveMP.rule(
            ManyPlus,
            (Val(:inputs), target_index),
            Marginalisation(),
            Val((:out, :inputs)),
            messages,
            nothing,
            nothing,
            nothing,
            ReactiveMP.AnnotationDict(),
            nothing,
        )
    end

    function manyplus_test_normal(
        parameterisation, mean_value, variance_value, ::Type{T}
    ) where {T}
        mean_t = convert(T, mean_value)
        variance_t = convert(T, variance_value)

        if parameterisation === :mean_variance
            return NormalMeanVariance(mean_t, variance_t)
        elseif parameterisation === :mean_precision
            return NormalMeanPrecision(mean_t, inv(variance_t))
        elseif parameterisation === :weighted_mean_precision
            precision = inv(variance_t)
            return NormalWeightedMeanPrecision(mean_t * precision, precision)
        elseif parameterisation === :normal
            return Normal(mean_t, sqrt(variance_t))
        end

        error("Unknown Normal parameterisation $(parameterisation).")
    end

    @testset "rules" begin
        parameterisations = (
            :mean_variance, :mean_precision, :weighted_mean_precision, :normal
        )

        for arity in (2, 3, 7)
            means = [(-1.0)^index * (index + 0.25) for index in 1:arity]
            variances = [0.2 + index / 3 for index in 1:arity]
            inputs = [
                manyplus_test_normal(
                    parameterisations[mod1(index, length(parameterisations))],
                    means[index],
                    variances[index],
                    Float64,
                ) for index in 1:arity
            ]

            forward = manyplus_rule_output(inputs)
            @test forward isa NormalMeanVariance
            @test collect(mean_var(forward)) ≈ [sum(means), sum(variances)]

            output = NormalWeightedMeanPrecision(4.0, 2.0)
            output_mean, output_variance = mean_var(output)
            for target_index in eachindex(inputs)
                other_indices = filter(!=(target_index), eachindex(inputs))
                other_inputs = inputs[other_indices]
                backward = manyplus_rule_input(
                    output, other_inputs, target_index
                )

                @test backward isa NormalMeanVariance
                @test collect(mean_var(backward)) ≈ [
                    output_mean - sum(means[other_indices]),
                    output_variance + sum(variances[other_indices]),
                ]
            end
        end

        float32_inputs = [
            NormalMeanVariance(Float32(0.5), Float32(0.25)),
            NormalMeanPrecision(Float32(-0.25), Float32(2.0)),
        ]
        float32_output = manyplus_rule_output(float32_inputs)
        @test float32_output isa NormalMeanVariance{Float32}

        promoted_output = manyplus_rule_output((
            float32_inputs[1], NormalMeanVariance(0.25, 0.75)
        ))
        @test promoted_output isa NormalMeanVariance{Float64}

        big_output = manyplus_rule_output((
            NormalMeanVariance(big"0.5", big"0.25"), float32_inputs[2]
        ))
        @test big_output isa NormalMeanVariance{BigFloat}

        input1 = NormalMeanPrecision(-0.4, 2.0)
        input2 = NormalWeightedMeanPrecision(0.3, 1.5)
        output = NormalMeanVariance(1.2, 0.7)

        manyplus_forward = manyplus_rule_output((input1, input2))
        binary_forward = @call_rule typeof(+)(:out, Marginalisation) (
            m_in1 = input1, m_in2 = input2
        )
        @test collect(mean_var(manyplus_forward)) ≈
            collect(mean_var(binary_forward))

        manyplus_backward1 = manyplus_rule_input(output, (input2,), 1)
        binary_backward1 = @call_rule typeof(+)(:in1, Marginalisation) (
            m_out = output, m_in2 = input2
        )
        @test collect(mean_var(manyplus_backward1)) ≈
            collect(mean_var(binary_backward1))

        manyplus_backward2 = manyplus_rule_input(output, (input1,), 2)
        binary_backward2 = @call_rule typeof(+)(:in2, Marginalisation) (
            m_out = output, m_in1 = input1
        )
        @test collect(mean_var(manyplus_backward2)) ≈
            collect(mean_var(binary_backward2))
    end

    @testset "Observed output" begin
        output = PointMass(5.0)
        for arity in (2, 3, 7)
            inputs = [
                manyplus_test_normal(
                    (
                        :mean_variance,
                        :mean_precision,
                        :weighted_mean_precision,
                        :normal,
                    )[mod1(i, 4)],
                    i / 2,
                    i / 4,
                    Float64,
                ) for i in 1:arity
            ]
            for k in eachindex(inputs)
                others = inputs[filter(!=(k), eachindex(inputs))]
                result = manyplus_rule_input(output, others, k)
                @test result isa NormalMeanVariance
                @test mean(result) ≈ 5 - sum(mean, others)
                @test var(result) ≈ sum(var, others)
            end
        end

        for parameterisation in
            (:mean_variance, :mean_precision, :weighted_mean_precision, :normal)
            gaussian = manyplus_test_normal(
                parameterisation, 0.5, 0.25, Float64
            )
            binary = @call_rule typeof(+)(:in1, Marginalisation) (
                m_out = output, m_in2 = gaussian
            )
            @test collect(
                mean_var(manyplus_rule_input(output, (gaussian,), 1))
            ) ≈ collect(mean_var(binary))
        end

        for (constant, T) in (
                (PointMass(2), Float32),
                (PointMass(2.0f0), Float32),
                (PointMass(2.0), Float64),
                (PointMass(big"2"), BigFloat),
            ),
            others in (
                (constant, NormalMeanVariance(0.5f0, 0.25f0)),
                (NormalMeanVariance(0.5f0, 0.25f0), constant),
            )

            result = manyplus_rule_input(PointMass(5.0f0), others, 2)
            @test result isa NormalMeanVariance{T}
            @test mean_var(result) == (T(2.5), T(0.25))
        end

        for (output, others, expected) in (
            (PointMass(5), (PointMass(2), PointMass(-1)), PointMass(4)),
            (PointMass(5.0f0), (PointMass(2.0f0),), PointMass(3.0f0)),
            (PointMass(5.0f0), (PointMass(2.0),), PointMass(3.0)),
            (PointMass(5), (PointMass(big"2"),), PointMass(big"3")),
        )
            result = manyplus_rule_input(output, others, 1)
            @test result isa typeof(expected)
            @test mean_var(result) == mean_var(expected)
        end
    end

    @testset "Constant inputs" begin
        inputs = [
            NormalMeanVariance(0.5, 0.25),
            PointMass(2),
            NormalMeanPrecision(-0.25, 2.0),
            PointMass(-1.0),
        ]
        for shift in 0:3
            ordered = circshift(inputs, shift)
            forward = manyplus_rule_output(ordered)
            @test forward isa NormalMeanVariance
            @test mean_var(forward) == (1.25, 0.75)
            for k in eachindex(ordered)
                other_inputs = ordered[filter(!=(k), eachindex(ordered))]
                backward = manyplus_rule_input(
                    NormalMeanVariance(5.0, 1.25), other_inputs, k
                )
                @test mean(backward) ≈ 3.75 + mean(ordered[k])
                @test var(backward) ≈ 2.0 - var(ordered[k])
            end
        end

        for parameterisation in
            (:mean_variance, :mean_precision, :weighted_mean_precision, :normal)
            gaussian = manyplus_test_normal(
                parameterisation, 0.5, 0.25, Float64
            )
            constant = PointMass(-2.0)
            for inputs in ((gaussian, constant), (constant, gaussian))
                expected = @call_rule typeof(+)(:out, Marginalisation) (
                    m_in1 = inputs[1], m_in2 = inputs[2]
                )
                @test collect(mean_var(manyplus_rule_output(inputs))) ≈
                    collect(mean_var(expected))
            end
            output = NormalMeanVariance(1.5, 0.5)
            expected = @call_rule typeof(+)(:in1, Marginalisation) (
                m_out = output, m_in2 = constant
            )
            @test collect(
                mean_var(manyplus_rule_input(output, (constant,), 1))
            ) ≈ collect(mean_var(expected))
        end

        for (inputs, expected) in (
            ((PointMass(2), PointMass(-3), PointMass(4)), PointMass(3)),
            ((PointMass(2.0f0), PointMass(-0.5f0)), PointMass(1.5f0)),
            ((PointMass(2), PointMass(-0.5)), PointMass(1.5)),
            ((PointMass(big"2"), PointMass(-0.5f0)), PointMass(big"1.5")),
        )
            result = manyplus_rule_output(inputs)
            @test result isa typeof(expected)
            @test mean_var(result) == mean_var(expected)
        end

        gaussian = NormalMeanVariance(0.5f0, 0.25f0)
        output = NormalMeanVariance(5.0f0, 1.0f0)
        for (constant, T) in (
                (PointMass(2), Float32),
                (PointMass(2.0f0), Float32),
                (PointMass(2.0), Float64),
                (PointMass(big"2"), BigFloat),
            ),
            inputs in ((constant, gaussian), (gaussian, constant))

            forward = manyplus_rule_output(inputs)
            @test forward isa NormalMeanVariance{T}
            @test mean_var(forward) == (T(2.5), T(0.25))
            backward = manyplus_rule_input(output, inputs, 1)
            @test backward isa NormalMeanVariance{T}
            @test mean_var(backward) == (T(2.5), T(1.25))
        end
    end
end
