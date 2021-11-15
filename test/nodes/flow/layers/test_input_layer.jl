module FlowNodeLayersInputLayerTest

using Test
using ReactiveMP 
using LinearAlgebra

using ReactiveMP: getdim

@testset "Input Layer" begin

    @testset "Constructor" begin

        # check for placeholder creation
        layer = InputLayer(2)
        @test typeof(layer)   == InputLayer
        @test typeof(layer)   <: ReactiveMP.AbstractLayerPlaceholder

        # check error handling
        @test_throws AssertionError InputLayer(1) 
        @test_throws AssertionError InputLayer(-1) 

    end

    @testset "Get" begin
        
        # check for layer
        for k = 2:10
            layer = InputLayer(k)
            @test layer.dim     == getdim(layer)
            @test getdim(layer) == k
        end

    end

    @testset "Integration in flow model" begin
        
        model = FlowModel((InputLayer(5), AdditiveCouplingLayer(PlanarFlow()), AdditiveCouplingLayer(PlanarFlow(); permute=false)))
        compiled_model = compile(model)
        @test typeof(first(getlayers(compiled_model))) <: AdditiveCouplingLayer
        for layer in getlayers(compiled_model)
            @test getdim(layer) == 5
        end

    end

end

end