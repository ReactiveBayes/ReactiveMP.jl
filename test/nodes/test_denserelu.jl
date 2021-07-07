module DenseReLUNodeTest

using Test
using ReactiveMP
using Random

@testset "DenseReLUNode" begin

    @testset "Creation" begin

        # univariate check without meta
        node = make_node(DenseReLU{1})

        @test functionalform(node)          === DenseReLU{1}
        @test sdtype(node)                  === Stochastic()
        @test name.(interfaces(node))       === (:output, :input, :w, :z, :f)
        @test factorisation(node)           === MeanField()
        @test metadata(node)                === nothing

        # univariate check with random meta data
        node = make_node(DenseReLU{1}, meta = 1)

        @test functionalform(node)          === DenseReLU{1}
        @test sdtype(node)                  === Stochastic()
        @test name.(interfaces(node))       === (:output, :input, :w, :z, :f)
        @test factorisation(node)           === MeanField()
        @test metadata(node)                === 1

        # multivariate check with common meta data
        meta = DenseReLUMeta(2)
        node = make_node(DenseReLU{2}, meta = meta)

        @test functionalform(node)          === DenseReLU{2}
        @test sdtype(node)                  === Stochastic()
        @test name.(interfaces(node))       === (:output, :input, :w, :w, :z, :z, :f, :f)
        @test factorisation(node)           === MeanField()
        @test metadata(node)                === meta
        
        # check when dimensions of node and meta data do not coincide
        @test_throws AssertionError make_node(DenseReLU{1}, meta = DenseReLUMeta(2))

    end

    @testset "Meta data" begin
        
        # check when no output dimension is given
        @test_throws MethodError DenseReLUMeta()

        # check with varying output dimension
        for k = 1:10
            meta = DenseReLUMeta(k)

            @test ReactiveMP.getdimout(meta)       === k
            @test ReactiveMP.getC(meta)            === 1.0
            @test ReactiveMP.getβ(meta)            === 10.0
            @test ReactiveMP.getγ(meta)            === 10.0
            for ki = 1:k
                @test ReactiveMP.getξk(meta,ki)    === 1.0
            end
            @test ReactiveMP.getusebias(meta)      === false
        end

        # check for changing defaults (C)
        for k = 1:10
            meta = DenseReLUMeta(k; C = 15)

            @test ReactiveMP.getdimout(meta)       === k
            @test ReactiveMP.getC(meta)            === 15.0
            @test ReactiveMP.getβ(meta)            === 10.0
            @test ReactiveMP.getγ(meta)            === 10.0
            for ki = 1:k
                @test ReactiveMP.getξk(meta,ki)    === 1.0
            end
            @test ReactiveMP.getusebias(meta)      === false
        end

        # check for changing defaults (β)
        for k = 1:10
            meta = DenseReLUMeta(k; β=100)

            @test ReactiveMP.getdimout(meta)       === k
            @test ReactiveMP.getC(meta)            === 1.0
            @test ReactiveMP.getβ(meta)            === 100.0
            @test ReactiveMP.getγ(meta)            === 10.0
            for ki = 1:k
                @test ReactiveMP.getξk(meta,ki)    === 1.0
            end
            @test ReactiveMP.getusebias(meta)      === false
        end

       # check for changing defaults (γ)
        for k = 1:10
            meta = DenseReLUMeta(k; γ=100.0)

            @test ReactiveMP.getdimout(meta)       === k
            @test ReactiveMP.getC(meta)            === 1.0
            @test ReactiveMP.getβ(meta)            === 10.0
            @test ReactiveMP.getγ(meta)            === 100.0
            for ki = 1:k
                @test ReactiveMP.getξk(meta,ki)    === 1.0
            end
            @test ReactiveMP.getusebias(meta)      === false
        end
        
        # check for changing defaults (use_bias)
        for k = 1:10
            meta = DenseReLUMeta(k; use_bias=true)

            @test ReactiveMP.getdimout(meta)       === k
            @test ReactiveMP.getC(meta)            === 1.0
            @test ReactiveMP.getβ(meta)            === 10.0
            @test ReactiveMP.getγ(meta)            === 10.0
            for ki = 1:k
                @test ReactiveMP.getξk(meta,ki)    === 1.0
            end
            @test ReactiveMP.getusebias(meta)      === true
        end

        # check for incorrect use_bias flag
        @test_throws TypeError DenseReLUMeta(2; use_bias=5)

    end

end

end
