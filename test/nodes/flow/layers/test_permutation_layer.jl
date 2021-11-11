module FlowNodeLayersPermutationLayerTest

using Test
using ReactiveMP 
using LinearAlgebra

@testset "Permutation Layer" begin

    @testset "Constructor" begin

        # check for placeholder creation
        layer = PermutationLayer()
        @test typeof(layer)   == ReactiveMP.PermutationLayerPlaceholder

        # check for layer with available P matrix
        P = PermutationMatrix(5)
        layer = PermutationLayer(5, P)
        @test layer.P == P
        @test typeof(layer)   <: PermutationLayer
        @test typeof(layer)   <: ReactiveMP.AbstractLayer
        @test typeof(layer.P) <: PermutationMatrix

        # check for layer with unknown P matrix
        layer = PermutationLayer(4)
        @test size(layer.P)   == (4, 4)
        @test typeof(layer)   <: PermutationLayer
        @test typeof(layer)   <: ReactiveMP.AbstractLayer
        @test typeof(layer.P) <: PermutationMatrix

    end

    @testset "Get" begin
        
        # check for layer with available P matrix
        P = PermutationMatrix(5)
        layer = PermutationLayer(5, P)
        @test layer.P           == getP(layer)
        @test layer.P           == getmat(layer)
        @test getP(layer)       == P
        @test layer.dim         == getdim(layer)
        @test getdim(layer)     == 5
        @test typeof(layer.P)   <: PermutationMatrix

        # check for layer with unknown P matrix
        layer = PermutationLayer(4)
        @test layer.P           == getP(layer)
        @test layer.P           == getmat(layer)
        @test size(layer.P)     == (4, 4)
        @test getdim(layer)     == layer.dim
        @test getdim(layer)     == 4
        @test typeof(layer.P)   <: PermutationMatrix

    end

    @testset "Prepare-Compile" begin
        
        layer = ReactiveMP._prepare(3, ReactiveMP.PermutationLayerPlaceholder())
        @test typeof(layer)     <: Tuple
        @test typeof(layer[1])  <: PermutationLayer
        @test layer[1].dim      == 3
        @test size(layer[1].P)  == (3,3)

        P = PermutationMatrix(4)
        layer = PermutationLayer(4, P)
        @test compile(layer)  == layer
        @test_throws ArgumentError compile(layer, 1)
        
        @test nr_params(layer) == 0

    end

    @testset "Base" begin
        
        # check for layer with available P matrix
        P = PermutationMatrix(5)
        layer = PermutationLayer(5, P)
        @test eltype(layer) <: Integer

        # check for layer with unknown P matrix
        layer = PermutationLayer(4)
        @test eltype(layer) <: Integer

    end

    @testset "Forward-Backward" begin
        
        # check forward function
        P = PermutationMatrix(3)
        layer = PermutationLayer(3, P)
        @test forward(layer, [5.0, 1.5, 8.5])  == P*[5.0, 1.5, 8.5]
        @test forward(layer, [4.0, 2.5, -9.0]) == P*[4.0, 2.5, -9.0]
        @test forward.(layer, [[5.0, 1.5, 8.5], [4.0, 2.5, -9.0]]) == [P*[5.0, 1.5, 8.5], P*[4.0, 2.5, -9.0]]

        # check forward! function
        P = PermutationMatrix(3)
        layer = PermutationLayer(3, P)
        output = zeros(3)
        forward!(output, layer, [5.0, 1.5, 8.5]) 
        @test output == P*[5.0, 1.5, 8.5]
        forward!(output, layer, [4.0, 2.5, -9.0]) 
        @test output == P*[4.0, 2.5, -9.0]

        # check bakward function
        P = PermutationMatrix(3)
        layer = PermutationLayer(3, P)
        @test backward(layer, [5.0, 1.5, 8.5])  == P'*[5.0, 1.5, 8.5]
        @test backward(layer, [4.0, 2.5, -9.0]) == P'*[4.0, 2.5, -9.0]
        @test backward.(layer, [[5.0, 1.5, 8.5], [4.0, 2.5, -9.0]]) == [P'*[5.0, 1.5, 8.5], P'*[4.0, 2.5, -9.0]]

        # check backward! function
        P = PermutationMatrix(3)
        layer = PermutationLayer(3, P)
        input = zeros(3)
        backward!(input, layer, [5.0, 1.5, 8.5]) 
        @test input == P'*[5.0, 1.5, 8.5]
        backward!(input, layer, [4.0, 2.5, -9.0]) 
        @test input == P'*[4.0, 2.5, -9.0]

    end

    @testset "Jacobian" begin
        
        # check jacobian function
        P = PermutationMatrix(3)
        layer = PermutationLayer(3, P)
        @test jacobian(layer, randn(3)) == P
        @test jacobian(layer, randn(3)) == P
        @test jacobian.(layer, [randn(3), randn(3)]) == [P, P]

        # check invjacobian function
        P = PermutationMatrix(3)
        layer = PermutationLayer(3, P)
        @test inv_jacobian(layer, randn(3)) == P'
        @test inv_jacobian(layer, randn(3)) == P'
        @test inv_jacobian.(layer, [randn(3), randn(3)]) == [P', P']

    end

    @testset "Utility Jacobian" begin
        
        # check utility functions jacobian 
        P = PermutationMatrix(3)
        layer = PermutationLayer(3, P)
        @test det_jacobian(layer, randn(3)) == det(P)
        @test det_jacobian(layer) == det(P)
        @test absdet_jacobian(layer, randn(3)) == 1.0
        @test absdet_jacobian(layer) == 1.0
        @test logdet_jacobian(layer, randn(3)) == 0.0
        @test logdet_jacobian(layer) == 0.0
        @test logabsdet_jacobian(layer, randn(3)) == 0.0
        @test logabsdet_jacobian(layer) == 0.0

    end

end

end