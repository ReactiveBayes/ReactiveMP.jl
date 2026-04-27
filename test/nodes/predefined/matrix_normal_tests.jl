
@testitem "MatrixNormalNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily, Distributions

    import ReactiveMP: alias_interface

    @testset "Node is registered" begin
        @test ReactiveMP.is_predefined_node(MatrixNormal) isa
            ReactiveMP.PredefinedNodeFunctionalForm
        @test ReactiveMP.sdtype(MatrixNormal) === ReactiveMP.Stochastic()
        @test ReactiveMP.interfaces(MatrixNormal) === Val((:out, :M, :U, :V))
    end

    @testset "Interface aliases" begin
        @test alias_interface(MatrixNormal, 2, :mean) === :M
        @test alias_interface(MatrixNormal, 3, :rowcov) === :U
        @test alias_interface(MatrixNormal, 4, :colcov) === :V
    end
end
