# BIFM's rules under BIFMSmoother. v6's meta was a cache: its rule towards `zprev` wrote the
# backward quantities, its rule towards `znext` wrote `μu`/`Σu`, and the rules towards `in`, `out`
# and `znext` read them back. The port recomputes them from the messages, so its forward rules
# take `m[:in]` and `m[:znext]` as well.

# From v6's `test/rules/bifm/zprev_tests.jl`, whose rule computed the cache from its messages and
# so ports as it is: v6's inputs, values, tolerances and promotion settings. v6's meta also held
# a cache (`H = [5 0; 0 4]`, ...) that its rule overwrote; it has no counterpart. The last two
# cases are not v6's: their values are ReactiveMP 6.5.0's, computed as described below.
@testitem "rules:BIFM:zprev" tags = [:rules] begin
    using BIFMMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions

    @testset "v6's table" begin
        algorithm = BIFMSmoother([2.0 0; 0 1], [3.0 0; 0 2], [4.0 0; 0 3])
        @test_message_update_rule(
            node = BIFM, target = :zprev, algorithm = algorithm,
            atol = Dict(Float32 => 1.0e-2, Float64 => 1.0e-2, BigFloat => 1.0e-8),
            cases = [
                (m = (out = MvNormalMeanPrecision([1, 2], [2 0; 0 1]), in = MvNormalMeanPrecision([1, 2], [1 0; 0 2]), znext = MvNormalMeanPrecision([1, 2], [1 0; 0 2])),) =>
                    MvNormalWeightedMeanPrecision([-0.60402684563758, -1.4782608695652169], [0.44295302013420557 0.0; 0.0 0.4782608695652171]),
                (m = (out = MvNormalMeanPrecision([3, 4], [6 0; 0 1]), in = MvNormalMeanPrecision([1, 6], [2 0; 0 2]), znext = MvNormalMeanPrecision([8, 2], [2 0; 0 2])),) =>
                    MvNormalWeightedMeanPrecision([-0.9321266968325688, -5.043478260869566], [0.886877828054339 0.0; 0.0 0.4782608695652171]),
                (m = (out = MvNormalMeanPrecision([5, 6], [2 0; 0 5]), in = MvNormalMeanPrecision([6, 2], [1 0; 0 1]), znext = MvNormalMeanPrecision([1, 9], [1 0; 0 1])),) =>
                    MvNormalWeightedMeanPrecision([-3.7114093959731633, -0.45945945945943834], [0.44295302013420557 0.0; 0.0 0.24864864864865122]),
            ],
        )
    end

    # For the Float64 tolerance of these two, see `BIFMForwardCases.ATOL` below.
    @testset "non-diagonal, a 3-dimensional state, a 2-dimensional input and output" begin
        algorithm = BIFMSmoother([0.9 0.1 0.0; -0.2 0.8 0.1; 0.05 0.0 0.95], [1.0 0.0; 0.5 1.0; 0.0 0.3], [1.0 0.5 0.0; 0.0 1.0 -0.4])
        m_out = MvNormalMeanCovariance([0.5, -1.0], [1.0 0.3; 0.3 2.0])
        m_in = MvNormalMeanCovariance([0.2, 0.1], [2.0 -0.4; -0.4 1.5])
        m_znext = MvNormalWeightedMeanPrecision([0.3, -0.2, 0.1], [0.5 0.1 0.0; 0.1 0.8 -0.2; 0.0 -0.2 0.6])
        @test_message_update_rule(
            node = BIFM, target = :zprev, algorithm = algorithm,
            atol = Dict(Float32 => 1.0e-3, Float64 => 1.0e-5, BigFloat => 1.0e-10),
            cases = [
                (m = (out = m_out, in = m_in, znext = m_znext),) => MvNormalWeightedMeanPrecision(
                    [0.3823068165691548, -0.3294331282526149, 0.2634066897660184],
                    [0.3828318426313889 -0.11566234704345414 0.16041410646009036; -0.1156623470434542 0.3092652164695752 -0.14791262089815266; 0.16041410646009036 -0.14791262089815266 0.5588320041228872],
                ),
            ],
        )
    end

    @testset "a scalar input and output, B not square" begin
        algorithm = BIFMSmoother([1.0 0.1; 0.0 1.0], reshape([0.005, 0.1], 2, 1), [1.0 0.0])
        m_out = MvNormalMeanPrecision([2.0], [4.0;;])
        m_in = MvNormalMeanCovariance([0.0], [1.0;;])
        m_znext = MvNormalMeanPrecision([2.2, 0.4], [2.0 0.5; 0.5 1.0])
        @test_message_update_rule(
            node = BIFM, target = :zprev, algorithm = algorithm,
            atol = Dict(Float32 => 1.0e-3, Float64 => 1.0e-5, BigFloat => 1.0e-10),
            cases = [
                (m = (out = m_out, in = m_in, znext = m_znext),) =>
                    MvNormalWeightedMeanPrecision([12.583139563647158, 2.736711522287637], [5.993667441745411 1.0912531539108494; 1.0912531539108494 1.1479184188393607]),
            ],
        )
    end
end

# The forward rules. v6's tables for them filled the meta's cache with numbers inconsistent with
# their messages (`H = [5 0; 0 4]`, `Λz = [7 0; 0 6]`, ...), which a stateless rule cannot
# reproduce, so these tables are new. Their expected values are ReactiveMP 6.5.0's, pasted as
# literals: in `compat/v6-comparison`, for each case and each target, a fresh
# `ReactiveMP.BIFMMeta(A, B, C)` was filled the way v6's schedule filled it, by v6's rule towards
# `zprev` on (out, in, znext) and then v6's rule towards `znext` on (out, in, zprev), which sets
# `μu`/`Σu` from m_in, and then v6's rule under test was run on it.
#
# The first three cases are v6's input sets: `m_out`, `m_in` and `m_znext` of v6's `zprev` table,
# and `m_zprev` of v6's `in` table. The fourth has non-diagonal matrices and messages with a
# 3-dimensional state and a 2-dimensional input and output; the fifth a 2-dimensional state and
# a scalar input and output.
@testmodule BIFMForwardCases begin
    using BIFMMessagePassingRules, BayesBase, ExponentialFamily, Distributions

    const DIAGONAL = BIFMSmoother([2.0 0; 0 1], [3.0 0; 0 2], [4.0 0; 0 3])
    const DENSE = BIFMSmoother([0.9 0.1 0.0; -0.2 0.8 0.1; 0.05 0.0 0.95], [1.0 0.0; 0.5 1.0; 0.0 0.3], [1.0 0.5 0.0; 0.0 1.0 -0.4])
    const SCALAR_INPUT = BIFMSmoother([1.0 0.1; 0.0 1.0], reshape([0.005, 0.1], 2, 1), [1.0 0.0])

    const INPUTS = [
        (
            DIAGONAL,
            (
                out = MvNormalMeanPrecision([1, 2], [2 0; 0 1]), in = MvNormalMeanPrecision([1, 2], [1 0; 0 2]),
                zprev = TerminalProdArgument(MvNormalMeanPrecision([1, 2], [1 0; 0 2])), znext = MvNormalMeanPrecision([1, 2], [1 0; 0 2]),
            ),
        ),
        (
            DIAGONAL,
            (
                out = MvNormalMeanPrecision([3, 4], [6 0; 0 1]), in = MvNormalMeanPrecision([1, 6], [2 0; 0 2]),
                zprev = TerminalProdArgument(MvNormalMeanPrecision([1, 6], [2 0; 0 2])), znext = MvNormalMeanPrecision([8, 2], [2 0; 0 2]),
            ),
        ),
        (
            DIAGONAL,
            (
                out = MvNormalMeanPrecision([5, 6], [2 0; 0 5]), in = MvNormalMeanPrecision([6, 2], [1 0; 0 1]),
                zprev = TerminalProdArgument(MvNormalMeanPrecision([6, 2], [1 0; 0 1])), znext = MvNormalMeanPrecision([1, 9], [1 0; 0 1]),
            ),
        ),
        (
            DENSE,
            (
                out = MvNormalMeanCovariance([0.5, -1.0], [1.0 0.3; 0.3 2.0]), in = MvNormalMeanCovariance([0.2, 0.1], [2.0 -0.4; -0.4 1.5]),
                zprev = TerminalProdArgument(MvNormalMeanCovariance([1.0, 0.0, -0.5], [1.5 0.2 0.1; 0.2 1.0 0.3; 0.1 0.3 2.0])),
                znext = MvNormalWeightedMeanPrecision([0.3, -0.2, 0.1], [0.5 0.1 0.0; 0.1 0.8 -0.2; 0.0 -0.2 0.6]),
            ),
        ),
        (
            SCALAR_INPUT,
            (
                out = MvNormalMeanPrecision([2.0], [4.0;;]), in = MvNormalMeanCovariance([0.0], [1.0;;]),
                zprev = TerminalProdArgument(MvNormalMeanCovariance([1.0, 0.5], [0.5 0.1; 0.1 0.3])), znext = MvNormalMeanPrecision([2.2, 0.4], [2.0 0.5; 0.5 1.0]),
            ),
        ),
    ]

    # The tolerance is chosen by the output's float type. A promotion that converts one input to
    # Float32 still gives a Float64 output, from an input rounded to Float32, so the Float64
    # tolerance covers that rounding, about 1e-7 here; v6's own tables allowed 1e-2 for both.
    const ATOL = Dict(Float32 => 1.0e-3, Float64 => 1.0e-5, BigFloat => 1.0e-10)

    # One table per algorithm, the cases in the order of INPUTS.
    function tables(expected)
        return [(algorithm, [(m = m,) => TerminalProdArgument(value) for ((a, m), value) in zip(INPUTS, expected) if a === algorithm]) for algorithm in (DIAGONAL, DENSE, SCALAR_INPUT)]
    end
end

@testitem "rules:BIFM:in" tags = [:rules] setup = [BIFMForwardCases] begin
    using BIFMMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions

    expected = [
        MvNormalMeanCovariance([-0.5704697986576899, -0.43478260869565055], [0.44482230530156874 -0.0; -0.0 0.13610586011342168]),
        MvNormalMeanCovariance([-0.3642533936651806, -1.9130434782608674], [0.22234905100222918 -0.0; -0.0 0.13610586011342168]),
        MvNormalMeanCovariance([-3.553691275167594, 0.08648648648651847], [0.44482230530156874 -0.0; -0.0 0.2527100073045967]),
        MvNormalMeanCovariance([-0.09829306680497657, -0.16079921451387755], [1.1272533495900308 -0.4786318439565481; -0.478631843956548 0.9899659835509176]),
        MvNormalMeanCovariance([0.07693068817097906], [0.9979123561008422;;]),
    ]
    for (algorithm, cases) in BIFMForwardCases.tables(expected)
        @test_message_update_rule(node = BIFM, target = :in, algorithm = algorithm, atol = BIFMForwardCases.ATOL, cases = cases)
    end
end

@testitem "rules:BIFM:out" tags = [:rules] setup = [BIFMForwardCases] begin
    using BIFMMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions

    expected = [
        MvNormalMeanCovariance([1.154362416107313, 3.3913043478260914], [0.4839421647673528 0.0; 0.0 0.7911153119092627]),
        MvNormalMeanCovariance([3.628959276018197, 6.52173913043478], [0.16305972441186706 0.0; 0.0 0.7911153119092627]),
        MvNormalMeanCovariance([5.355704697986411, 6.518918918919172], [0.4839421647673528 0.0; 0.0 0.19485756026296563]),
        MvNormalMeanCovariance([0.5717340592368403, -0.2706498421747006], [0.6118244287017089 0.1092749140117092; 0.10927491401170916 0.6630586484563835]),
        MvNormalMeanCovariance([1.0503846534408552], [0.5224791109712239;;]),
    ]
    for (algorithm, cases) in BIFMForwardCases.tables(expected)
        @test_message_update_rule(node = BIFM, target = :out, algorithm = algorithm, atol = BIFMForwardCases.ATOL, cases = cases)
    end
end

@testitem "rules:BIFM:znext" tags = [:rules] setup = [BIFMForwardCases] begin
    using BIFMMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions

    expected = [
        MvNormalMeanCovariance([0.28859060402682823, 1.1304347826086971], [0.03024638529795955 0.0; 0.0 0.08790170132325141]),
        MvNormalMeanCovariance([0.9072398190045489, 2.1739130434782603], [0.010191232775741691 0.0; 0.0 0.08790170132325141]),
        MvNormalMeanCovariance([1.338926174496602, 2.1729729729730574], [0.03024638529795955 0.0; 0.0 0.021650840029218407]),
        MvNormalMeanCovariance(
            [0.8017069331950233, -0.45994574791636605, -0.4732397643541633],
            [0.6116928701469415 -0.16725793402547923 -0.04734373223195001; -0.16725793402547923 0.6695579703209864 0.38591815008042385; -0.04734373223195 0.3859181500804239 1.8889699887483513],
        ),
        MvNormalMeanCovariance([1.0503846534408552, 0.507693068817098], [0.5224791109712239 0.12483700594800047; 0.12483700594800048 0.30183584943049835]),
    ]
    for (algorithm, cases) in BIFMForwardCases.tables(expected)
        @test_message_update_rule(node = BIFM, target = :znext, algorithm = algorithm, atol = BIFMForwardCases.ATOL, cases = cases)
    end
end

@testitem "rules:BIFM:BIFMSmoother" tags = [:rules] begin
    using BIFMMessagePassingRules

    # The element types are promoted, and the matrices converted to `Matrix`.
    smoother = BIFMSmoother([1 0; 0 1], Float32[1 0; 0 1], [1.0 0.0])
    @test smoother isa BIFMSmoother{Float64}
    @test smoother.A isa Matrix{Float64} && smoother.B isa Matrix{Float64} && smoother.C isa Matrix{Float64}
    @test smoother.A == [1 0; 0 1] && smoother.B == [1 0; 0 1] && smoother.C == [1 0]
    @test BIFMSmoother(Float32[1 0; 0 1], Float32[1; 0;;], Float32[1 0]) isa BIFMSmoother{Float32}
    @test BIFMSmoother([1 0; 0 1], [1; 0;;], [1 0]) isa BIFMSmoother{Int}

    # Inconsistent sizes: a non-square A, a B with the wrong number of rows, a C with the wrong
    # number of columns. v6 asserted the second.
    @test_throws DimensionMismatch BIFMSmoother(ones(2, 3), ones(2, 2), ones(2, 2))
    @test_throws DimensionMismatch BIFMSmoother(ones(2, 2), ones(3, 2), ones(2, 2))
    @test_throws DimensionMismatch BIFMSmoother(ones(2, 2), ones(2, 2), ones(2, 3))
end

# v6's marginal rule returned the joint of `in` and `zprev` only, for a cluster over all three;
# the port throws, since the free energy of a BIFM model is not supported.
@testitem "rules:BIFM:marginal (:in, :zprev, :znext)" tags = [:rules] begin
    using BIFMMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions

    algorithm = BIFMSmoother([2.0 0; 0 1], [3.0 0; 0 2], [4.0 0; 0 3])
    m = (
        out = MvNormalMeanPrecision([1.0, 2.0], [2.0 0; 0 1]), in = MvNormalMeanPrecision([1.0, 2.0], [1.0 0; 0 2]),
        zprev = TerminalProdArgument(MvNormalMeanPrecision([1.0, 2.0], [1.0 0; 0 2])), znext = MvNormalMeanPrecision([1.0, 2.0], [1.0 0; 0 3]),
    )
    @test_throws BIFMMessagePassingRules.BIFMFreeEnergyError call_marginal_update_rule(BIFM, (:in, :zprev, :znext); m, algorithm)
    error = try
        call_marginal_update_rule(BIFM, (:in, :zprev, :znext); m, algorithm)
    catch e
        e
    end
    @test error.node === :BIFM
    @test occursin("`BIFM` is not supported", sprint(showerror, error))
end
