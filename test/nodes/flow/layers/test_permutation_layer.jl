module FlowNodeLayersPermutationLayerTest

using Test
using ReactiveMP 
using LinearAlgebra

@testset "Permutation Layer" begin

    @testset "Constructor" begin
        
        # check for layer with available P matrix
        P = PermutationMatrix(5)
        layer = PermutationLayer(P)
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
        layer = PermutationLayer(P)
        @test layer.P == getP(layer)
        @test layer.P == getmat(layer)
        @test typeof(layer.P) <: PermutationMatrix

        # check for layer with unknown P matrix
        layer = PermutationLayer(4)
        @test layer.P == getP(layer)
        @test layer.P == getmat(layer)
        @test size(layer.P)   == (4, 4)
        @test typeof(layer.P) <: PermutationMatrix

    end

    @testset "Base" begin
        
        # check for layer with available P matrix
        P = PermutationMatrix(5)
        layer = PermutationLayer(P)
        @test eltype(layer) <: Int

        # check for layer with unknown P matrix
        layer = PermutationLayer(4)
        @test eltype(layer) <: Int

    end

    @testset "Forward-Backward" begin
        
        # check forward function
        P = PermutationMatrix(3)
        layer = PermutationLayer(P)
        @test forward(layer, [5.0, 1.5, 8.5])  == P*[5.0, 1.5, 8.5]
        @test forward(layer, [4.0, 2.5, -9.0]) == P*[4.0, 2.5, -9.0]
        @test forward.(layer, [[5.0, 1.5, 8.5], [4.0, 2.5, -9.0]]) == [P*[5.0, 1.5, 8.5], P*[4.0, 2.5, -9.0]]

        # check forward! function
        P = PermutationMatrix(3)
        layer = PermutationLayer(P)
        output = zeros(3)
        forward!(output, layer, [5.0, 1.5, 8.5]) 
        @test output == P*[5.0, 1.5, 8.5]
        forward!(output, layer, [4.0, 2.5, -9.0]) 
        @test output == P*[4.0, 2.5, -9.0]

        # check bakward function
        P = PermutationMatrix(3)
        layer = PermutationLayer(P)
        @test backward(layer, [5.0, 1.5, 8.5])  == P'*[5.0, 1.5, 8.5]
        @test backward(layer, [4.0, 2.5, -9.0]) == P'*[4.0, 2.5, -9.0]
        @test backward.(layer, [[5.0, 1.5, 8.5], [4.0, 2.5, -9.0]]) == [P'*[5.0, 1.5, 8.5], P'*[4.0, 2.5, -9.0]]

        # check backward! function
        P = PermutationMatrix(3)
        layer = PermutationLayer(P)
        input = zeros(3)
        backward!(input, layer, [5.0, 1.5, 8.5]) 
        @test input == P'*[5.0, 1.5, 8.5]
        backward!(input, layer, [4.0, 2.5, -9.0]) 
        @test input == P'*[4.0, 2.5, -9.0]

    end

    @testset "Jacobian" begin
        
        # check jacobian function
        P = PermutationMatrix(3)
        layer = PermutationLayer(P)
        @test jacobian(layer, randn(3)) == P
        @test jacobian(layer, randn(3)) == P
        @test jacobian.(layer, [randn(3), randn(3)]) == [P, P]

        # check invjacobian function
        P = PermutationMatrix(3)
        layer = PermutationLayer(P)
        @test inv_jacobian(layer, randn(3)) == P'
        @test inv_jacobian(layer, randn(3)) == P'
        @test inv_jacobian.(layer, [randn(3), randn(3)]) == [P', P']

    end

    @testset "Utility Jacobian" begin
        
        # check utility functions jacobian 
        P = PermutationMatrix(3)
        layer = PermutationLayer(P)
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