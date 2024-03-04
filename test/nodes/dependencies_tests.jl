@testitem "collect_latest_messages" begin
    include("../testutilities.jl")
    using BayesBase

    import ReactiveMP: NodeInterface, collect_latest_messages, getdata, getrecent

    @testset let (tag, stream) = collect_latest_messages(())
        @test tag === nothing
        @test check_stream_updated_once(stream) === nothing
    end

    a = NodeInterface(:a, ConstVariable(1))
    b = NodeInterface(:b, ConstVariable(2))
    c = NodeInterface(:c, ConstVariable(3))

    @testset let (tag, stream) = collect_latest_messages((a, b, c))
        @test tag === Val((:a, :b, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(2), PointMass(3))
    end

    @testset let (tag, stream) = collect_latest_messages((a, b))
        @test tag === Val((:a, :b))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(2))
    end

    @testset let (tag, stream) = collect_latest_messages((b, c))
        @test tag === Val((:b, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(2), PointMass(3))
    end

    @testset let (tag, stream) = collect_latest_messages((a, c))
        @test tag === Val((:a, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(3))
    end
end

@testitem "collect_latest_marginals" begin
    include("../testutilities.jl")
    using BayesBase

    import ReactiveMP: NodeInterface, FactorNodeLocalMarginal, getmarginal, collect_latest_marginals, getdata, getrecent

    struct ArbitraryNode end

    a = FactorNodeLocalMarginal(:a)
    b = FactorNodeLocalMarginal(:b)
    c = FactorNodeLocalMarginal(:c)

    setmarginal!(a, getmarginal(ConstVariable(1), IncludeAll()))
    setmarginal!(b, getmarginal(ConstVariable(2), IncludeAll()))
    setmarginal!(c, getmarginal(ConstVariable(3), IncludeAll()))

    @testset let (tag, stream) = collect_latest_marginals((a, b, c))
        @test tag === Val((:a, :b, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(2), PointMass(3))
    end

    @testset let (tag, stream) = collect_latest_marginals((a, b))
        @test tag === Val((:a, :b))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(2))
    end

    @testset let (tag, stream) = collect_latest_marginals((b, c))
        @test tag === Val((:b, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(2), PointMass(3))
    end

    @testset let (tag, stream) = collect_latest_marginals((a, c))
        @test tag === Val((:a, :c))
        @test getdata.(getrecent.(check_stream_updated_once(stream))) === (PointMass(1), PointMass(3))
    end
end