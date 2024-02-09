
@testitem "FlowNode" begin
    using ReactiveMP
    using ReactiveMP: getL, getα, getβ, getκ, getλ

    # test for basics in the flow.jl file
    @testset "Basics" begin

        # tests for node creation
        @testset "Creation" begin

            # create example Flow node
            node = make_node(Flow, FactorNodeCreationOptions(nothing, 1, nothing))

            # check whether Flow node creation has been a success
            @test functionalform(node) === Flow
            @test sdtype(node) === Deterministic()
            @test name.(interfaces(node)) === (:out, :in)
            @test factorisation(node) === ((1, 2),)
            @test localmarginalnames(node) === (:out_in,)
            @test metadata(node) === 1
        end

        # tests for meta data
        @testset "MetaData" begin
            @testset "Default" begin

                # create example meta data
                meta = FlowMeta(compile(FlowModel(2, (AdditiveCouplingLayer(PlanarFlow()),))))

                # check whether the Flow node has a default metadata structure
                @test_throws ErrorException ReactiveMP.default_meta(Flow)

                # check whether the getmodel function works
                @test typeof(getmodel(meta)) <: ReactiveMP.AbstractCompiledFlowModel
                @test getmodel(meta) == meta.model
                @test getapproximation(meta) == meta.approximation
                @test getapproximation(meta) isa Linearization
            end

            @testset "Linearization" begin

                # create example meta data
                meta = FlowMeta(compile(FlowModel(2, (AdditiveCouplingLayer(PlanarFlow()),))), Linearization())

                # check whether the Flow node has a default metadata structure
                @test_throws ErrorException ReactiveMP.default_meta(Flow)

                # check whether the getmodel function works
                @test typeof(getmodel(meta)) <: ReactiveMP.AbstractCompiledFlowModel
                @test getmodel(meta) == meta.model
                @test getapproximation(meta) == meta.approximation
                @test getapproximation(meta) isa Linearization
            end

            @testset "Unscented" begin

                # create example meta data
                meta = FlowMeta(compile(FlowModel(2, (AdditiveCouplingLayer(PlanarFlow()),))), Unscented(3))

                # check whether the Flow node has a default metadata structure
                @test_throws ErrorException ReactiveMP.default_meta(Flow)

                # check whether the getmodel function works
                @test typeof(getmodel(meta)) <: ReactiveMP.AbstractCompiledFlowModel
                @test getmodel(meta) == meta.model
                @test getapproximation(meta) == meta.approximation
                @test getapproximation(meta) isa Unscented
                @test getL(getapproximation(meta)) == 3
                @test getα(getapproximation(meta)) == 1e-3
                @test getβ(getapproximation(meta)) == 2.0
                @test getκ(getapproximation(meta)) == 0.0
                @test getλ(getapproximation(meta)) == 3e-6 - 3
            end
        end
    end
end
