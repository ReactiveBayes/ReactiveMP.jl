module FlowNodeTest

using Test
using ReactiveMP 

include("../../test_helpers.jl")

using .ReactiveMPTestingHelpers

@testset "FlowNode" begin

    # test for basics in the flow.jl file
    @testset "Basics" begin

        # tests for node creation
        @testset "Creation" begin

            # create example Flow node
            node = make_node(Flow, meta = 1)
    
            # check whether Flow node creation has been a success
            @test functionalform(node)          === Flow
            @test sdtype(node)                  === Deterministic()
            @test name.(interfaces(node))       === (:out, :in)
            @test factorisation(node)           === ((1, 2), )
            @test localmarginalnames(node)      === (:out_in, )
            @test metadata(node)                === 1

        end

        # tests for meta data
        @testset "MetaData" begin

            @testset "Default" begin

                # create example meta data
                meta = FlowMeta(FlowModel((NiceLayer(PlanarMap()),)))
                
                # check whether the Flow node has a default metadata structure
                @test_throws ErrorException ReactiveMP.default_meta(Flow)

                # check whether the getmodel function works
                @test typeof(getmodel(meta)) <: ReactiveMP.AbstractFlowModel
                @test getmodel(meta) == meta.model
                @test getapproximation(meta) == meta.approximation
                @test getapproximation(meta) == Linearization()
            end

            @testset "Linearization" begin
                
                # create example meta data
                meta = FlowMeta(FlowModel((NiceLayer(PlanarMap()),)), Linearization())
                
                # check whether the Flow node has a default metadata structure
                @test_throws ErrorException ReactiveMP.default_meta(Flow)

                # check whether the getmodel function works
                @test typeof(getmodel(meta)) <: ReactiveMP.AbstractFlowModel
                @test getmodel(meta) == meta.model
                @test getapproximation(meta) == meta.approximation
                @test typeof(getapproximation(meta)) == Linearization

            end

            @testset "Unscented" begin
                
                # create example meta data
                meta = FlowMeta(FlowModel((NiceLayer(PlanarMap()),)), Unscented(3))
                
                # check whether the Flow node has a default metadata structure
                @test_throws ErrorException ReactiveMP.default_meta(Flow)

                # check whether the getmodel function works
                @test typeof(getmodel(meta)) <: ReactiveMP.AbstractFlowModel
                @test getmodel(meta) == meta.model
                @test getapproximation(meta) == meta.approximation
                @test typeof(getapproximation(meta)) == Unscented
                @test getL(getapproximation(meta))   == 3
                @test getα(getapproximation(meta))   == 1e-3
                @test getβ(getapproximation(meta))   == 2.0
                @test getκ(getapproximation(meta))   == 0.0
                @test getλ(getapproximation(meta))   == 3e-6 - 3

            end
        
        end
        
    end

    # test for model specifics in the flow_models folder
    @testset "Models" begin

        addtests("flow_models/test_flow_model.jl")
        
    end

    # test for layer specifics in the flow_layers folder
    @testset "Layers" begin
        
        addtests("flow_layers/test_nice_layer.jl")
        addtests("flow_layers/test_reverse_nice_layer.jl")

    end

    # test for neural network specifics in the neural_networks folder
    @testset "NeuralNetworks" begin
        
        addtests("neural_networks/test_planar_map.jl")

    end

end

end
