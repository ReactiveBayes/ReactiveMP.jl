
module ReactiveMPLoggerPipelineStageTest

using ReactiveMP
using Distributions
using Rocket

import ReactiveMP: tag

@testitem "LoggerPipelineStage" begin
    struct DummyNode end

    @node DummyNode Stochastic [out, x, y]

    # In real applications the stream should be a stream of messages
    # For testing purposes it does not really matter though
    stream = Subject(String)
    node = make_node(DummyNode)

    @testset "no prefix" begin
        io = IOBuffer()
        pipeline = LoggerPipelineStage(io)
        modified_stream = apply_pipeline_stage(pipeline, node, tag(first(interfaces(node))), stream)
        subscription = subscribe!(modified_stream, void())

        next!(stream, "hello")

        logged_str = String(take!(io))

        @test contains(logged_str, "Log") # default prefix
        @test contains(logged_str, "DummyNode")
        @test contains(logged_str, "out")
        @test contains(logged_str, "hello")

        unsubscribe!(subscription)
    end

    @testset "with custom prefix" begin
        io = IOBuffer()
        pipeline = LoggerPipelineStage(io, "custom_prefix")
        modified_stream = apply_pipeline_stage(pipeline, node, tag(first(interfaces(node))), stream)
        subscription = subscribe!(modified_stream, void())

        next!(stream, "hello")

        logged_str = String(take!(io))

        @test contains(logged_str, "custom_prefix")
        @test contains(logged_str, "DummyNode")
        @test contains(logged_str, "out")
        @test contains(logged_str, "hello")

        unsubscribe!(subscription)
    end
end

end
