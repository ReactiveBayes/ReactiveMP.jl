# The tensor algebra the rules are made of, and the axes each input covers.

@testitem "tensors:multiply and sum out" tags = [:rules] begin
    using DiscreteTransitionMessagePassingRules
    using DiscreteTransitionMessagePassingRules: multiply_dimensions!, sum_out_dimensions

    A = reshape(collect(1.0:24.0), 2, 3, 4)
    v, w = [0.5, 2.0, 1.0], [1.0, 0.0, 3.0, 2.0]

    # Along one axis, and along two with a matrix whose axes run along them.
    @test multiply_dimensions!(copy(A), (2,), v) == [A[i, j, k] * v[j] for i in 1:2, j in 1:3, k in 1:4]
    M = v * w'
    @test multiply_dimensions!(copy(A), (2, 3), M) == [A[i, j, k] * M[j, k] for i in 1:2, j in 1:3, k in 1:4]
    # `dims` are increasing, as every key's axes are: the values are reshaped, not permuted.
    @test issorted(DiscreteTransitionMessagePassingRules.discrete_transition_axes((:out, :in, (:T, 1), (:T, 3)), 4))

    # Summing out keeps the axes as singletons: an inner product over them.
    @test sum_out_dimensions(copy(A), (2,), v) == reshape([sum(A[i, j, k] * v[j] for j in 1:3) for i in 1:2, k in 1:4], 2, 1, 4)
    @test vec(sum_out_dimensions(copy(A), (2, 3), M)) == [sum(A[i, j, k] * M[j, k] for j in 1:3, k in 1:4) for i in 1:2]

    # In place for one element type; promoted, into a new tensor, for mixed ones.
    B = copy(A)
    @test multiply_dimensions!(B, (1,), [1.0, 2.0]) === B
    @test eltype(multiply_dimensions!(Float32.(A), (1,), [1.0, 2.0])) == Float64
end

@testitem "tensors:axes" tags = [:rules] begin
    using DiscreteTransitionMessagePassingRules
    using DiscreteTransitionMessagePassingRules: discrete_transition_axes

    # `out` is axis 1, `in` 2, the `k`-th `T` 2 + k, and a joint its members' axes in order.
    @test discrete_transition_axes(:out, 1) == (1,)
    @test discrete_transition_axes(:in, 1) == (2,)
    @test discrete_transition_axes((:T, 3), 1) == (5,)
    @test discrete_transition_axes((:out, (:T, 2)), 2) == (1, 4)
    @test discrete_transition_axes(((:T, 1), (:T, 3)), 2) == (3, 5)
    # A whole group spans the axes of the input's tensor left after the other members.
    @test discrete_transition_axes((:out, :in, :T), 4) == (1, 2, 3, 4)
    @test discrete_transition_axes((:T,), 3) == (3, 4, 5)
    @test_throws ArgumentError discrete_transition_axes(:a, 1)
    @test_throws ArgumentError discrete_transition_axes((:m, 1), 1)
end
