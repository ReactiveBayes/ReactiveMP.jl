module ReactiveMPFactorisationSpecTest 

using Test
using ReactiveMP 

import ReactiveMP: FunctionalIndex
import ReactiveMP: CombinedRange, SplittedRange, is_splitted
import ReactiveMP: __as_unit_range, __factorisation_specification_resolve_index
import ReactiveMP: resolve_factorisation
import ReactiveMP: DefaultConstraints

using GraphPPL # for `@constraints` macro

@testset "Factorisation constraints specification" begin 

    @testset "CombinedRange" begin 
        for left in 1:3, right in 10:13
            cr = CombinedRange(left, right)

            @test firstindex(cr) === left
            @test lastindex(cr)  === right
            @test !is_splitted(cr)
            
            for i in left:right
                @test i ∈ cr
                @test !((i + lastindex(cr) + 1) ∈ cr)
            end
        end
    end

    @testset "SplittedRange" begin 
        for left in 1:3, right in 10:13
            cr = SplittedRange(left, right)

            @test firstindex(cr) === left
            @test lastindex(cr)  === right
            @test is_splitted(cr)
            
            for i in left:right
                @test i ∈ cr
                @test !((i + lastindex(cr) + 1) ∈ cr)
            end
        end
    end

    @testset "__as_unit_range" begin    
        for i in 1:3
            __as_unit_range(i) === i:i
        end

        for i in 1:3
            __as_unit_range(CombinedRange(i, i + 1)) === (i):(i + 1)
            __as_unit_range(SplittedRange(i, i + 1)) === (i):(i + 1)
        end
    end

    @testset "__factorisation_specification_resolve_index" begin
        collection = randomvar(:x, 3)
        
        @test __factorisation_specification_resolve_index(nothing, randomvar(:x)) === nothing
        @test_throws ErrorException __factorisation_specification_resolve_index(1, randomvar(:x))
        @test_throws ErrorException __factorisation_specification_resolve_index(FunctionalIndex{:begin}(firstindex), randomvar(:x))
        
        @test __factorisation_specification_resolve_index(nothing, collection) === nothing
        @test __factorisation_specification_resolve_index(1, collection) === 1
        @test __factorisation_specification_resolve_index(3, collection) === 3
        
        @test_throws ErrorException __factorisation_specification_resolve_index(6, collection)
        
        @test __factorisation_specification_resolve_index(FunctionalIndex{:begin}(firstindex) + 1, collection) === 2
        @test __factorisation_specification_resolve_index(FunctionalIndex{:begin}(firstindex) + 1 + 1, collection) === 3
        @test __factorisation_specification_resolve_index(FunctionalIndex{:end}(lastindex) - 1, collection) === 2
        
        @test_throws ErrorException __factorisation_specification_resolve_index(FunctionalIndex{:begin}(firstindex) + 100, collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(FunctionalIndex{:end}(lastindex) - 100, collection)
        
        @test __factorisation_specification_resolve_index(CombinedRange(1, 3), collection) === CombinedRange(1, 3)
        @test __factorisation_specification_resolve_index(CombinedRange(1, 2), collection) === CombinedRange(1, 2)
        @test __factorisation_specification_resolve_index(CombinedRange(FunctionalIndex{:begin}(firstindex), 2), collection) === CombinedRange(1, 2)
        @test __factorisation_specification_resolve_index(CombinedRange(1, FunctionalIndex{:end}(lastindex)), collection) === CombinedRange(1, 3)
        @test __factorisation_specification_resolve_index(CombinedRange(FunctionalIndex{:begin}(firstindex) + 1, FunctionalIndex{:end}(lastindex) - 1), collection) === CombinedRange(2, 2)
        
        @test __factorisation_specification_resolve_index(SplittedRange(1, 3), collection) === SplittedRange(1, 3)
        @test __factorisation_specification_resolve_index(SplittedRange(1, 2), collection) === SplittedRange(1, 2)
        @test __factorisation_specification_resolve_index(SplittedRange(FunctionalIndex{:begin}(firstindex), 2), collection) === SplittedRange(1, 2)
        @test __factorisation_specification_resolve_index(SplittedRange(1, FunctionalIndex{:end}(lastindex)), collection) === SplittedRange(1, 3)
        @test __factorisation_specification_resolve_index(SplittedRange(FunctionalIndex{:begin}(firstindex) + 1, FunctionalIndex{:end}(lastindex) - 1), collection) === SplittedRange(2, 2)
        
        @test_throws ErrorException __factorisation_specification_resolve_index(CombinedRange(1, 5), collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(SplittedRange(1, 5), collection)
        
        @test_throws ErrorException __factorisation_specification_resolve_index(CombinedRange(FunctionalIndex{:begin}(firstindex) + 100, 2), collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(CombinedRange(1, FunctionalIndex{:end}(lastindex) - 100), collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(CombinedRange(FunctionalIndex{:begin}(firstindex) + 1, FunctionalIndex{:end}(lastindex) - 100), collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(CombinedRange(FunctionalIndex{:begin}(firstindex) + 100, FunctionalIndex{:end}(lastindex)), collection)
        
        @test_throws ErrorException __factorisation_specification_resolve_index(SplittedRange(FunctionalIndex{:begin}(firstindex) + 100, 2), collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(SplittedRange(1, FunctionalIndex{:end}(lastindex) - 100), collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(SplittedRange(FunctionalIndex{:begin}(firstindex) + 1, FunctionalIndex{:end}(lastindex) - 100), collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(SplittedRange(FunctionalIndex{:begin}(firstindex) + 100, FunctionalIndex{:end}(lastindex)), collection)
    end

    @testset "Factorisation constraints resolution" begin

        # Factorisation constrains resolution function accepts an expression as an input for error printing
        # We don't care about actual expression in tests
        expr = :(test ~ Test(test))

        @testset "Use case #1" begin
            cs = @constraints begin 
                q(x, y) = q(x)q(y)
            end

            model = Model()

            x = randomvar(model, :x)
            y = randomvar(model, :y)

            @test resolve_factorisation(expr, (x, y), cs, model) === ((1,), (2, ))
        end

        @testset "Use case #2" begin
            @constraints function cs2(flag)
                if flag
                    q(x, y) = q(x, y)
                else
                    q(x, y) = q(x)q(y)
                end
            end

            model = Model()

            x = randomvar(model, :x)
            y = randomvar(model, :y)

            @test resolve_factorisation(expr, (x, y), cs2(true), model) === ((1, 2), )
            @test resolve_factorisation(expr, (x, y), cs2(false), model) === ((1,), (2, ))
        end

        @testset "Use case #3" begin
            cs = @constraints begin
                q(x, y) = q(x)q(y)
            end

            model = Model()

            x = randomvar(model, :x, 10)
            y = randomvar(model, :y, 10)

            for i in 1:5
                @test resolve_factorisation(expr, (x[i], y[i]), cs, model) === ((1, ), (2, ))
                @test resolve_factorisation(expr, (x[i + 1], y[i]), cs, model) === ((1, ), (2, ))
                @test resolve_factorisation(expr, (x[i], y[i + 1]), cs, model) === ((1, ), (2, ))
                @test resolve_factorisation(expr, (y[i], x[i]), cs, model) === ((1, ), (2, ))
                @test resolve_factorisation(expr, (y[2], x[i]), cs, model) === ((1, ), (2, ))
                @test resolve_factorisation(expr, (y[i], x[i + 1]), cs, model) === ((1, ), (2, ))
                @test resolve_factorisation(expr, (x[i], x[i + 1], y[i], y[i + 1]), cs, model) === ((1, 2), (3, 4))
                @test resolve_factorisation(expr, (y[i], y[i + 1], x[i], x[i + 1]), cs, model) === ((1, 2), (3, 4))
                @test resolve_factorisation(expr, (y[i], x[i], y[i + 1], x[i + 1]), cs, model) === ((1, 3), (2, 4))
                @test resolve_factorisation(expr, (x[i], y[i], x[i + 1], y[i + 1]), cs, model) === ((1, 3), (2, 4))
                @test resolve_factorisation(expr, (x[i], y[i + 1], x[i + 1], y[i]), cs, model) === ((1, 3), (2, 4))
                @test resolve_factorisation(expr, (x[i], x[i + 1], x[i + 2], y[i], y[i + 1], y[i + 2]), cs, model) === ((1, 2, 3), (4, 5, 6))
                @test resolve_factorisation(expr, (y[i], y[i + 1], y[i + 2], x[i], x[i + 1], x[i + 2]), cs, model) === ((1, 2, 3), (4, 5, 6))
                @test resolve_factorisation(expr, (x[i], y[i], x[i + 1], y[i + 1], x[i + 2], y[i + 2]), cs, model) === ((1, 3, 5), (2, 4, 6))
                @test resolve_factorisation(expr, (y[i], x[i], y[i + 1], x[i + 1], y[i + 2], x[i + 2]), cs, model) === ((1, 3, 5), (2, 4, 6))
            end
        end

        @testset "Use case #4" begin
            @constraints function cs4(flag)
                if flag
                    q(x, y) = q(x)q(y)
                end
                q(x, y, z) = q(x, y)q(z)
            end

            model = Model()

            x = randomvar(model, :x, 10)
            y = randomvar(model, :y, 10)
            z = randomvar(model, :z)

            for i in 1:10
                @test resolve_factorisation(expr, (x[i], y[i]), cs4(true), model) === ((1, ), (2, ))
                @test resolve_factorisation(expr, (x[i], y[i], z), cs4(true), model) === ((1, ), (2, ), (3, ))
                @test resolve_factorisation(expr, (x[i], z, y[i]), cs4(true), model) === ((1, ), (2, ), (3, ))
                @test resolve_factorisation(expr, (z, x[i], y[i]), cs4(true), model) === ((1, ), (2, ), (3, ))
                @test resolve_factorisation(expr, (x[i], z), cs4(true), model) === ((1, ), (2, ), )
                @test resolve_factorisation(expr, (y[i], z), cs4(true), model) === ((1, ), (2, ), )

                @test resolve_factorisation(expr, (x[i], y[i]), cs4(false), model) === ((1, 2), )
                @test resolve_factorisation(expr, (x[i], y[i], z), cs4(false), model) === ((1, 2, ), (3, ))
                @test resolve_factorisation(expr, (x[i], z, y[i]), cs4(false), model) === ((1, 3, ), (2, ))
                @test resolve_factorisation(expr, (z, x[i], y[i]), cs4(false), model) === ((1, ), (2, 3, ))
                @test resolve_factorisation(expr, (z, y[i], x[i]), cs4(false), model) === ((1, ), (2, 3, ))
                @test resolve_factorisation(expr, (x[i], z), cs4(false), model) === ((1, ), (2, ), )
                @test resolve_factorisation(expr, (y[i], z), cs4(false), model) === ((1, ), (2, ), )
            end
            
        end

        @testset "Use case #5" begin
            cs = @constraints begin 

            end

            model = Model()

            x = randomvar(model, :x, 11)
            y = randomvar(model, :y, 11)
            z = randomvar(model, :z)
            
            for i in 1:10
                @test resolve_factorisation(expr, (x[i], y[i], z), cs, model) === ((1, 2, 3), )
                @test resolve_factorisation(expr, (z, x[i], y[i]), cs, model) === ((1, 2, 3), )
                @test resolve_factorisation(expr, (x[i], z, y[i]), cs, model) === ((1, 2, 3), )
                @test resolve_factorisation(expr, (x[i], x[i + 1], y[i], y[i + 1], z), cs, model) === ((1, 2, 3, 4, 5), )
                @test resolve_factorisation(expr, (z, x[i], x[i + 1], y[i], y[i + 1]), cs, model) === ((1, 2, 3, 4, 5), )
                @test resolve_factorisation(expr, (x[i], z, x[i + 1], y[i], y[i + 1]), cs, model) === ((1, 2, 3, 4, 5), )
                @test resolve_factorisation(expr, (x[i], x[i + 1], z, y[i], y[i + 1]), cs, model) === ((1, 2, 3, 4, 5), )
                @test resolve_factorisation(expr, (x[i], x[i + 1], y[i], z, y[i + 1]), cs, model) === ((1, 2, 3, 4, 5), )
                @test resolve_factorisation(expr, (x[i], x[i + 1], y[i], y[i + 1], z), cs, model) === ((1, 2, 3, 4, 5), )
            end
            
        end

        @testset "Use case #6" begin
            cs = @constraints begin 
                q(x) = q(x[begin])..q(x[end])
            end

            model = Model()

            x = randomvar(model, :x, 10)

            for i in 1:8
                @test resolve_factorisation(expr, (x[i], x[i + 1]), cs, model) === ((1,), (2, ))
                @test resolve_factorisation(expr, (x[i], x[i + 1], x[i + 2]), cs, model) === ((1,), (2, ), (3, ))
            end
        end

        @testset "Use case #7" begin
            cs = @constraints function cs6(n)
                q(x) = q(x[1:n])q(x[n + 1])..q(x[end])
            end

            model = Model()

            x = randomvar(model, :x, 10)
            z = randomvar(model, :z)
            
            for i in 1:10, n in 1:9
                @test resolve_factorisation(expr, (x[i], z), cs6(n), model) === ((1, 2), )
            end    
            @test resolve_factorisation(expr, (x[1], x[2]), cs6(5), model) === ((1, 2), )
            @test resolve_factorisation(expr, (x[2], x[1]), cs6(5), model) === ((1, 2), )
            @test resolve_factorisation(expr, (x[5], x[6]), cs6(5), model) === ((1, ), (2, ), )
            @test resolve_factorisation(expr, (x[6], x[5]), cs6(5), model) === ((1, ), (2, ), )
            
            @test_throws ReactiveMP.ClusterIntersectionError resolve_factorisation(expr, (x[1], x[2], z), cs6(1), model) 
            @test_throws ReactiveMP.ClusterIntersectionError resolve_factorisation(expr, (x[2], x[1], z), cs6(1), model) 
            @test_throws ReactiveMP.ClusterIntersectionError resolve_factorisation(expr, (z, x[1], x[2]), cs6(1), model) 
            @test_throws ReactiveMP.ClusterIntersectionError resolve_factorisation(expr, (x[1], z, x[2]), cs6(1), model) 

            for n in 2:9
                @test resolve_factorisation(expr, (x[1], x[2], z), cs6(n), model) === ((1, 2, 3), )
            end
        end

        @testset "Use case #8" begin
            @constraints function cs8(flag)
                q(x, y) = q(x[begin], y[begin])..q(x[end], y[end])   
                q(x, y, t) = q(x, y)q(t)
                q(x, y, r) = q(x, y)q(r)
                if flag
                    q(t, r) = q(t)q(r)
                end
            end
                    
            model = Model()
                    
            y = randomvar(model, :y, 10)
            x = randomvar(model, :x, 10)
            t = randomvar(model, :t)
            r = randomvar(model, :r)
            
            for i in 1:9
                @test ReactiveMP.resolve_factorisation(expr, (y[i], y[i + 1], x[i], x[i + 1], t, r), cs8(false), model) === ((1, 3), (2, 4), (5, 6, ))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], x[i + 1], y[i], y[i + 1], t, r), cs8(false), model) === ((1, 3), (2, 4), (5, 6, ))
                @test ReactiveMP.resolve_factorisation(expr, (t, r, x[i], x[i + 1], y[i], y[i + 1]), cs8(false), model) === ((1, 2), (3, 5), (4, 6, ))
                @test ReactiveMP.resolve_factorisation(expr, (t, x[i], x[i + 1], y[i], y[i + 1], r), cs8(false), model) === ((1, 6), (2, 4), (3, 5, ))
                @test ReactiveMP.resolve_factorisation(expr, (y[i], y[i + 1], x[i], x[i + 1], t, r), cs8(true), model) === ((1, 3), (2, 4), (5, ), (6, ))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], x[i + 1], y[i], y[i + 1], t, r), cs8(true), model) === ((1, 3), (2, 4), (5,), (6, ))
                @test ReactiveMP.resolve_factorisation(expr, (t, r, x[i], x[i + 1], y[i], y[i + 1]), cs8(true), model) === ((1,), (2, ), (3, 5), (4, 6, ))
                @test ReactiveMP.resolve_factorisation(expr, (t, x[i], x[i + 1], y[i], y[i + 1], r), cs8(true), model) === ((1,), (2, 4), (3, 5, ), (6, ))
            end
            
        end

        @testset "Use case #9" begin
            cs = @constraints begin
                q(x, y) = q(x)q(y)
                q(x, y, t, r) = q(x, y)q(t)q(r)
                q(x, w) = q(x)q(w)
                q(y, w) = q(y)q(w)
                q(x) = q(x[begin:begin+2])q(x[begin+3])..q(x[end])
            end
                    
            model = Model()

            x = randomvar(model, :x, 10)
            y = randomvar(model, :y, 10)
            t = randomvar(model, :t, 10)
            r = randomvar(model, :r)
        
            @test ReactiveMP.resolve_factorisation(expr, (x[1], x[2], y[1], t[1]), cs, model) === ((1, 2, ), (3, ), (4, ))
            @test ReactiveMP.resolve_factorisation(expr, (x[2], x[3], y[1], t[1]), cs, model) === ((1, 2, ), (3, ), (4, ))
            @test ReactiveMP.resolve_factorisation(expr, (x[2], x[3], x[4], y[1], t[1]), cs, model) === ((1, 2, ), (3, ), (4, ), (5, ))
            @test ReactiveMP.resolve_factorisation(expr, (x[3], x[4], y[1], t[1]), cs, model) === ((1, ), (2, ), (3, ), (4, ))
            @test ReactiveMP.resolve_factorisation(expr, (x[3], x[4], y[1], t[1], r), cs, model) === ((1, ), (2, ), (3, ), (4, ), (5, ))
            @test ReactiveMP.resolve_factorisation(expr, (x[2], x[3], x[4], y[1], t[1], r), cs, model) === ((1, 2, ), (3, ), (4, ), (5, ), (6, ))
        end

        @testset "Use case #10" begin
            cs = @constraints begin
                q(x, y) = (q(x[begin])..q(x[end]))*(q(y[begin])..q(y[end]))        
                q(x, y, t) = q(x, y)q(t)
                q(x, y, r) = q(x, y)q(r)
            end
                    
            model = Model()
                    
            y = randomvar(model, :y, 10)
            x = randomvar(model, :x, 10)
            t = randomvar(model, :t)
            r = randomvar(model, :r)

            for i in 1:9
                @test ReactiveMP.resolve_factorisation(expr, (y[i], y[i + 1], x[i], x[i + 1], t, r), cs, model) === ((1,) , (2, ), (3, ), (4, ), (5, 6, ))
                @test ReactiveMP.resolve_factorisation(expr, (y[i], x[i + 1], x[i], y[i + 1], t, r), cs, model) === ((1,) , (2, ), (3, ), (4, ), (5, 6, ))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], y[i + 1], y[i], x[i + 1], t, r), cs, model) === ((1,) , (2, ), (3, ), (4, ), (5, 6, ))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], x[i + 1], y[i], y[i + 1], t, r), cs, model) === ((1,) , (2, ), (3, ), (4, ), (5, 6, ))
                @test ReactiveMP.resolve_factorisation(expr, (r, y[i], y[i + 1], x[i], x[i + 1], t), cs, model) === ((1, 6) , (2, ), (3, ), (4, ), (5, ))
                @test ReactiveMP.resolve_factorisation(expr, (t, y[i], y[i + 1], x[i], x[i + 1], r), cs, model) === ((1, 6) , (2, ), (3, ), (4, ), (5, ))
                @test ReactiveMP.resolve_factorisation(expr, (r, t, y[i], y[i + 1], x[i], x[i + 1]), cs, model) === ((1, 2) , (3, ), (4, ), (5, ), (6, ))
                @test ReactiveMP.resolve_factorisation(expr, (t, r, y[i], y[i + 1], x[i], x[i + 1]), cs, model) === ((1, 2) , (3, ), (4, ), (5, ), (6, ))
            end
        end

        @testset "Use case #11" begin
            cs = @constraints begin
                q(x, y) = q(y[1])q(x[begin], y[begin+1])..q(x[end], y[end])       
                q(x, y, t) = q(x, y)q(t)
                q(x, y, r) = q(x, y)q(r)
            end
                    
            model = Model()
                    
            y = randomvar(model, :y, 11)
            x = randomvar(model, :x, 10)
            t = randomvar(model, :t)
            r = randomvar(model, :r)

            @test ReactiveMP.resolve_factorisation(expr, (x[1], y[2], x[2], y[3]), cs, model) === ((1, 2), (3, 4), )
            @test ReactiveMP.resolve_factorisation(expr, (x[1], y[2], x[2], y[3], y[1]), cs, model) === ((1, 2), (3, 4), (5, ))
            @test ReactiveMP.resolve_factorisation(expr, (y[1], y[2], x[1], x[2], t, r), cs, model) === ((1, ), (2, 3), (4, ), (5, 6))
            @test ReactiveMP.resolve_factorisation(expr, (y[1], y[2], y[3], x[1], x[2], x[3], t, r), cs, model) === ((1, ), (2, 4), (3, 5), (6,), (7, 8))
        end

        @testset "Use case #12" begin 
            cs = @constraints begin
                q(x, y) = q(x[begin])*q(x[begin+1:end])*q(y)
            end
                    
            model = Model()
                    
            y = randomvar(model, :y, 11)
            x = randomvar(model, :x, 10)

            
            for i in 1:10
            if i > 1 && i < 10
                @test ReactiveMP.resolve_factorisation(expr, (x[1], x[i]), cs, model) === ((1,), (2,))
                @test ReactiveMP.resolve_factorisation(expr, (x[1], x[i], x[i + 1]), cs, model) === ((1,), (2, 3))
            end
            @test ReactiveMP.resolve_factorisation(expr, (x[1], x[2], x[3], y[i]), cs, model) === ((1,), (2, 3), (4, ))
            @test ReactiveMP.resolve_factorisation(expr, (x[1], x[2], x[3], x[4], y[i]), cs, model) === ((1,), (2, 3, 4), (5, ))
            end
             
        end

        @testset "Use case #13" begin 
            cs = @constraints begin 
                q(x, y) = q(x)q(y)
            end

            model = Model()

            x = randomvar(model, :x)
            y = randomvar(model, :y)
            tmp = randomvar(model, :tmp, proxy_variables = (y, ))

            @test ReactiveMP.resolve_factorisation(expr, (x, y), cs, model) === ((1, ), (2, ))
            @test ReactiveMP.resolve_factorisation(expr, (x, tmp), cs, model) === ((1, ), (2, ))
        end

        @testset "Use case #14" begin 
            # Check proxy vars
            @constraints function cs14(flag)
                q(x, y) = q(x)q(y)
                if flag 
                    q(x, y, z) = q(x)q(y, z)
                else
                    q(x, y, z) = q(y)q(x, z)
                end
            end

            model = Model()

            x = randomvar(model, :x, 10)
            y = randomvar(model, :y, 10)
            z = randomvar(model, :z)

            d = datavar(model, :d, Float64, 10)
            c = constvar(model, :c, (i) -> i, 10)

            # different proxy vars
            tmp1 = Vector{RandomVariable}(undef, 10)
            tmp2 = Vector{RandomVariable}(undef, 10)
            tmp3 = Vector{RandomVariable}(undef, 10)
            tmp4 = Vector{RandomVariable}(undef, 10)
            tmp5 = Vector{RandomVariable}(undef, 10)
            tmp6 = Vector{RandomVariable}(undef, 10)
            tmp7 = Vector{RandomVariable}(undef, 10)

            for i in 1:10
                tmp1[i] = randomvar(model, :tmp1, proxy_variables = (y[i], ))
                tmp2[i] = randomvar(model, :tmp2, proxy_variables = (y[i], d[i]))
                tmp3[i] = randomvar(model, :tmp3, proxy_variables = (y[i], c[i]))
                tmp4[i] = randomvar(model, :tmp2, proxy_variables = (c[i], y[i], d[i]))
                tmp5[i] = randomvar(model, :tmp3, proxy_variables = (d[i], y[i], c[i]))
                tmp6[i] = randomvar(model, :tmp2, proxy_variables = (d[i], y[i]))
                tmp7[i] = randomvar(model, :tmp3, proxy_variables = (c[i], y[i]))
            end

            for i in 1:10
                @test ReactiveMP.resolve_factorisation(expr, (x[i], y[i], z), cs14(true), model) === ((1, ), (2, 3))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], tmp1[i], z), cs14(true), model) === ((1, ), (2, 3))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], tmp2[i], z), cs14(true), model) === ((1, ), (2, 3))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], tmp3[i], z), cs14(true), model) === ((1, ), (2, 3))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], tmp4[i], z), cs14(true), model) === ((1, ), (2, 3))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], tmp5[i], z), cs14(true), model) === ((1, ), (2, 3))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], tmp6[i], z), cs14(true), model) === ((1, ), (2, 3))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], tmp7[i], z), cs14(true), model) === ((1, ), (2, 3))

                @test ReactiveMP.resolve_factorisation(expr, (x[i], y[i], z), cs14(false), model) === ((1, 3), (2, ))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], tmp1[i], z), cs14(false), model) === ((1, 3), (2, ))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], tmp2[i], z), cs14(false), model) === ((1, 3), (2, ))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], tmp3[i], z), cs14(false), model) === ((1, 3), (2, ))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], tmp4[i], z), cs14(false), model) === ((1, 3), (2, ))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], tmp5[i], z), cs14(false), model) === ((1, 3), (2, ))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], tmp6[i], z), cs14(false), model) === ((1, 3), (2, ))
                @test ReactiveMP.resolve_factorisation(expr, (x[i], tmp7[i], z), cs14(false), model) === ((1, 3), (2, ))
            end
        end

        @testset "Use case #15" begin
            # empty and default constraints still should factorize out datavar and constvar
            empty = @constraints begin 
                # empty
            end

            for cs in (empty, DefaultConstraints)
                let model = Model()
                    d = datavar(model, :d, Float64)
                    c = constvar(model, :c, 1.0)
                    x = randomvar(model, :x)
                    y = randomvar(model, :y)

                    @test ReactiveMP.resolve_factorisation(expr, (d, d), cs, model) === ((1,), (2, ))
                    @test ReactiveMP.resolve_factorisation(expr, (c, c), cs, model) === ((1,), (2, ))
                    @test ReactiveMP.resolve_factorisation(expr, (d, x), cs, model) === ((1,), (2, ))
                    @test ReactiveMP.resolve_factorisation(expr, (d, x, y), cs, model) === ((1,), (2, 3))
                    @test ReactiveMP.resolve_factorisation(expr, (x, d, y), cs, model) === ((1, 3), (2, ))
                    @test ReactiveMP.resolve_factorisation(expr, (x, y, d), cs, model) === ((1, 2), (3, ))
                    @test ReactiveMP.resolve_factorisation(expr, (c, x), cs, model) === ((1,), (2, ))
                    @test ReactiveMP.resolve_factorisation(expr, (c, x, y), cs, model) === ((1,), (2, 3))
                    @test ReactiveMP.resolve_factorisation(expr, (x, c, y), cs, model) === ((1, 3), (2, ))
                    @test ReactiveMP.resolve_factorisation(expr, (x, y, c), cs, model) === ((1, 2), (3, ))
                    @test ReactiveMP.resolve_factorisation(expr, (c, d), cs, model) === ((1,), (2, ))
                    @test ReactiveMP.resolve_factorisation(expr, (c, x, d), cs, model) === ((1,), (2, ), (3, ))
                    @test ReactiveMP.resolve_factorisation(expr, (x, c, d), cs, model) === ((1, ), (2, ), (3, ))
                    @test ReactiveMP.resolve_factorisation(expr, (x, d, c), cs, model) === ((1, ), (2,), (3, ))
                    @test ReactiveMP.resolve_factorisation(expr, (c, x, d, y), cs, model) === ((1,), (2, 4), (3, ))
                    @test ReactiveMP.resolve_factorisation(expr, (x, c, d, y), cs, model) === ((1, 4), (2, ), (3, ))
                    @test ReactiveMP.resolve_factorisation(expr, (x, d, c, y), cs, model) === ((1, 4), (2,), (3, ))
                    @test ReactiveMP.resolve_factorisation(expr, (c, x, y, d), cs, model) === ((1,), (2, 3), (4, ))
                    @test ReactiveMP.resolve_factorisation(expr, (x, c, y, d), cs, model) === ((1, 3), (2, ), (4, ))
                    @test ReactiveMP.resolve_factorisation(expr, (x, d, y, c), cs, model) === ((1, 3), (2,), (4, ))
                end
            end
        end

        ## Error testing below

        @testset "Error case #1" begin 
            # Names are not unique
            @test_throws ErrorException @constraints begin
                q(x, y) = q(x[begin], x[begin])..q(x[end], x[end])q(y) 
            end
        end

        @testset "Error case #2" begin 
            @test_throws LoadError eval(Meta.parse("""
                @constraints begin
                    q(x, y) = q(x) # `y` is not present
                end
            """))

            @test_throws LoadError eval(Meta.parse("""
                @constraints begin
                    q(x, y) = q(y) # `x` is not present
                end
            """))

            @test_throws LoadError eval(Meta.parse("""
                @constraints begin
                    q(x, y) = q(t) # `t` is not unknown
                end
            """))

        end

        @testset "Error case #3" begin 
            # Redefinition
            @test_throws ErrorException @constraints begin
                q(x, y) = q(x, y)
                q(x, y) = q(x)q(y)
            end

            @constraints function ercs3(flag)
                q(x, y) = q(x, y)
                if flag
                    q(x, y) = q(x)q(y)
                end
            end

            @test ercs3(false) isa ReactiveMP.ConstraintsSpecification
            @test_throws ErrorException ercs3(true)
        end

        @testset "Error case #4" begin 
            # multiple proxy vars
            cs = @constraints begin
                q(x, y, z) = q(x)q(y)q(z)
            end

            model = Model()

            x = randomvar(model, :x)
            y = randomvar(model, :y)

            z = randomvar(model, :z)
            d = datavar(model, :d, Float64)
            c = constvar(model, :c, 1)

            tmp1 = randomvar(model, :tmp, proxy_variables = (x, y))
            tmp2 = randomvar(model, :tmp, proxy_variables = (x, y, d))
            tmp3 = randomvar(model, :tmp, proxy_variables = (x, y, c))
            tmp4 = randomvar(model, :tmp, proxy_variables = (d, x, y))
            tmp5 = randomvar(model, :tmp, proxy_variables = (x, c, y))
            tmp6 = randomvar(model, :tmp, proxy_variables = (x, d, y))
            tmp7 = randomvar(model, :tmp, proxy_variables = (c, x, y))
            tmp8 = randomvar(model, :tmp, proxy_variables = (c, x, y, d))
            tmp9 = randomvar(model, :tmp, proxy_variables = (d, x, y, c))
            
            @test_throws ErrorException resolve_factorisation(expr, (z, tmp1), cs, model)
            @test_throws ErrorException resolve_factorisation(expr, (z, tmp2), cs, model)
            @test_throws ErrorException resolve_factorisation(expr, (z, tmp3), cs, model)
            @test_throws ErrorException resolve_factorisation(expr, (z, tmp4), cs, model)
            @test_throws ErrorException resolve_factorisation(expr, (z, tmp5), cs, model)
            @test_throws ErrorException resolve_factorisation(expr, (z, tmp6), cs, model)
            @test_throws ErrorException resolve_factorisation(expr, (z, tmp7), cs, model)
            @test_throws ErrorException resolve_factorisation(expr, (z, tmp8), cs, model)
            @test_throws ErrorException resolve_factorisation(expr, (z, tmp9), cs, model)
        end

    end

end

end