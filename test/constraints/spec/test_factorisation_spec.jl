module ReactiveMPFactorisationSpecTest

using Test
using ReactiveMP
using Logging

import ReactiveMP: FunctionalIndex
import ReactiveMP: CombinedRange, SplittedRange, is_splitted
import ReactiveMP: __as_unit_range, __factorisation_specification_resolve_index
import ReactiveMP: resolve_factorisation
import ReactiveMP: DefaultConstraints, UnspecifiedConstraints
import ReactiveMP: setanonymous!, activate!

using GraphPPL # for `@constraints` macro

@testset "Factorisation constraints specification" begin
    @testset "CombinedRange" begin
        for left in 1:3, right in 10:13
            cr = CombinedRange(left, right)

            @test firstindex(cr) === left
            @test lastindex(cr) === right
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
            @test lastindex(cr) === right
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
            __as_unit_range(CombinedRange(i, i + 1)) === (i):(i+1)
            __as_unit_range(SplittedRange(i, i + 1)) === (i):(i+1)
        end
    end

    @testset "__factorisation_specification_resolve_index" begin
        collection = randomvar(:x, 3)

        @test __factorisation_specification_resolve_index(nothing, randomvar(:x)) === nothing
        @test_throws ErrorException __factorisation_specification_resolve_index(1, randomvar(:x))
        @test_throws ErrorException __factorisation_specification_resolve_index(
            FunctionalIndex{:begin}(firstindex),
            randomvar(:x)
        )

        @test __factorisation_specification_resolve_index(nothing, collection) === nothing
        @test __factorisation_specification_resolve_index(1, collection) === 1
        @test __factorisation_specification_resolve_index(3, collection) === 3

        @test_throws ErrorException __factorisation_specification_resolve_index(6, collection)

        @test __factorisation_specification_resolve_index(FunctionalIndex{:begin}(firstindex) + 1, collection) === 2
        @test __factorisation_specification_resolve_index(FunctionalIndex{:begin}(firstindex) + 1 + 1, collection) === 3
        @test __factorisation_specification_resolve_index(FunctionalIndex{:end}(lastindex) - 1, collection) === 2

        @test_throws ErrorException __factorisation_specification_resolve_index(
            FunctionalIndex{:begin}(firstindex) + 100,
            collection
        )
        @test_throws ErrorException __factorisation_specification_resolve_index(
            FunctionalIndex{:end}(lastindex) - 100,
            collection
        )

        @test __factorisation_specification_resolve_index(CombinedRange(1, 3), collection) === CombinedRange(1, 3)
        @test __factorisation_specification_resolve_index(CombinedRange(1, 2), collection) === CombinedRange(1, 2)
        @test __factorisation_specification_resolve_index(
            CombinedRange(FunctionalIndex{:begin}(firstindex), 2),
            collection
        ) === CombinedRange(1, 2)
        @test __factorisation_specification_resolve_index(
            CombinedRange(1, FunctionalIndex{:end}(lastindex)),
            collection
        ) === CombinedRange(1, 3)
        @test __factorisation_specification_resolve_index(
            CombinedRange(FunctionalIndex{:begin}(firstindex) + 1, FunctionalIndex{:end}(lastindex) - 1),
            collection
        ) === CombinedRange(2, 2)

        @test __factorisation_specification_resolve_index(SplittedRange(1, 3), collection) === SplittedRange(1, 3)
        @test __factorisation_specification_resolve_index(SplittedRange(1, 2), collection) === SplittedRange(1, 2)
        @test __factorisation_specification_resolve_index(
            SplittedRange(FunctionalIndex{:begin}(firstindex), 2),
            collection
        ) === SplittedRange(1, 2)
        @test __factorisation_specification_resolve_index(
            SplittedRange(1, FunctionalIndex{:end}(lastindex)),
            collection
        ) === SplittedRange(1, 3)
        @test __factorisation_specification_resolve_index(
            SplittedRange(FunctionalIndex{:begin}(firstindex) + 1, FunctionalIndex{:end}(lastindex) - 1),
            collection
        ) === SplittedRange(2, 2)

        @test_throws ErrorException __factorisation_specification_resolve_index(CombinedRange(1, 5), collection)
        @test_throws ErrorException __factorisation_specification_resolve_index(SplittedRange(1, 5), collection)

        @test_throws ErrorException __factorisation_specification_resolve_index(
            CombinedRange(FunctionalIndex{:begin}(firstindex) + 100, 2),
            collection
        )
        @test_throws ErrorException __factorisation_specification_resolve_index(
            CombinedRange(1, FunctionalIndex{:end}(lastindex) - 100),
            collection
        )
        @test_throws ErrorException __factorisation_specification_resolve_index(
            CombinedRange(FunctionalIndex{:begin}(firstindex) + 1, FunctionalIndex{:end}(lastindex) - 100),
            collection
        )
        @test_throws ErrorException __factorisation_specification_resolve_index(
            CombinedRange(FunctionalIndex{:begin}(firstindex) + 100, FunctionalIndex{:end}(lastindex)),
            collection
        )

        @test_throws ErrorException __factorisation_specification_resolve_index(
            SplittedRange(FunctionalIndex{:begin}(firstindex) + 100, 2),
            collection
        )
        @test_throws ErrorException __factorisation_specification_resolve_index(
            SplittedRange(1, FunctionalIndex{:end}(lastindex) - 100),
            collection
        )
        @test_throws ErrorException __factorisation_specification_resolve_index(
            SplittedRange(FunctionalIndex{:begin}(firstindex) + 1, FunctionalIndex{:end}(lastindex) - 100),
            collection
        )
        @test_throws ErrorException __factorisation_specification_resolve_index(
            SplittedRange(FunctionalIndex{:begin}(firstindex) + 100, FunctionalIndex{:end}(lastindex)),
            collection
        )
    end

    @testset "Factorisation constraints resolution" begin

        # Factorisation constrains resolution function accepts a `fform` symbol as an input for error printing
        # We don't care about actual symbol in tests
        struct TestFactorisationStochastic end
        struct TestFactorisationDeterministic end

        ReactiveMP.sdtype(::Type{TestFactorisationStochastic})    = ReactiveMP.Stochastic()
        ReactiveMP.sdtype(::Type{TestFactorisationDeterministic}) = ReactiveMP.Deterministic()

        fform = TestFactorisationStochastic

        @testset "Use case #1" begin
            cs = @constraints begin
                q(x, y) = q(x)q(y)
            end

            model = FactorGraphModel()

            x = randomvar(model, :x)
            y = randomvar(model, :y)

            @test resolve_factorisation(cs, model, fform, (x, y)) === ((1,), (2,))
        end

        @testset "Use case #2" begin
            @constraints function cs2(flag)
                if flag
                    q(x, y) = q(x, y)
                else
                    q(x, y) = q(x)q(y)
                end
            end

            model = FactorGraphModel()

            x = randomvar(model, :x)
            y = randomvar(model, :y)

            @test resolve_factorisation(cs2(true), model, fform, (x, y)) === ((1, 2),)
            @test resolve_factorisation(cs2(false), model, fform, (x, y)) === ((1,), (2,))
        end

        @testset "Use case #3" begin
            cs = @constraints begin
                q(x, y) = q(x)q(y)
            end

            model = FactorGraphModel()

            x = randomvar(model, :x, 10)
            y = randomvar(model, :y, 10)

            for i in 1:5
                @test resolve_factorisation(cs, model, fform, (x[i], y[i])) === ((1,), (2,))
                @test resolve_factorisation(cs, model, fform, (x[i+1], y[i])) === ((1,), (2,))
                @test resolve_factorisation(cs, model, fform, (x[i], y[i+1])) === ((1,), (2,))
                @test resolve_factorisation(cs, model, fform, (y[i], x[i])) === ((1,), (2,))
                @test resolve_factorisation(cs, model, fform, (y[2], x[i])) === ((1,), (2,))
                @test resolve_factorisation(cs, model, fform, (y[i], x[i+1])) === ((1,), (2,))
                @test resolve_factorisation(cs, model, fform, (x[i], x[i+1], y[i], y[i+1])) === ((1, 2), (3, 4))
                @test resolve_factorisation(cs, model, fform, (y[i], y[i+1], x[i], x[i+1])) === ((1, 2), (3, 4))
                @test resolve_factorisation(cs, model, fform, (y[i], x[i], y[i+1], x[i+1])) === ((1, 3), (2, 4))
                @test resolve_factorisation(cs, model, fform, (x[i], y[i], x[i+1], y[i+1])) === ((1, 3), (2, 4))
                @test resolve_factorisation(cs, model, fform, (x[i], y[i+1], x[i+1], y[i])) === ((1, 3), (2, 4))
                @test resolve_factorisation(cs, model, fform, (x[i], x[i+1], x[i+2], y[i], y[i+1], y[i+2])) ===
                      ((1, 2, 3), (4, 5, 6))
                @test resolve_factorisation(cs, model, fform, (y[i], y[i+1], y[i+2], x[i], x[i+1], x[i+2])) ===
                      ((1, 2, 3), (4, 5, 6))
                @test resolve_factorisation(cs, model, fform, (x[i], y[i], x[i+1], y[i+1], x[i+2], y[i+2])) ===
                      ((1, 3, 5), (2, 4, 6))
                @test resolve_factorisation(cs, model, fform, (y[i], x[i], y[i+1], x[i+1], y[i+2], x[i+2])) ===
                      ((1, 3, 5), (2, 4, 6))
            end
        end

        @testset "Use case #4" begin
            @constraints function cs4(flag)
                if flag
                    q(x, y) = q(x)q(y)
                end
                q(x, y, z) = q(x, y)q(z)
            end

            model = FactorGraphModel()

            x = randomvar(model, :x, 10)
            y = randomvar(model, :y, 10)
            z = randomvar(model, :z)

            for i in 1:10
                @test resolve_factorisation(cs4(true), model, fform, (x[i], y[i])) === ((1,), (2,))
                @test resolve_factorisation(cs4(true), model, fform, (x[i], y[i], z)) === ((1,), (2,), (3,))
                @test resolve_factorisation(cs4(true), model, fform, (x[i], z, y[i])) === ((1,), (2,), (3,))
                @test resolve_factorisation(cs4(true), model, fform, (z, x[i], y[i])) === ((1,), (2,), (3,))
                @test resolve_factorisation(cs4(true), model, fform, (x[i], z)) === ((1,), (2,))
                @test resolve_factorisation(cs4(true), model, fform, (y[i], z)) === ((1,), (2,))

                @test resolve_factorisation(cs4(false), model, fform, (x[i], y[i])) === ((1, 2),)
                @test resolve_factorisation(cs4(false), model, fform, (x[i], y[i], z)) === ((1, 2), (3,))
                @test resolve_factorisation(cs4(false), model, fform, (x[i], z, y[i])) === ((1, 3), (2,))
                @test resolve_factorisation(cs4(false), model, fform, (z, x[i], y[i])) === ((1,), (2, 3))
                @test resolve_factorisation(cs4(false), model, fform, (z, y[i], x[i])) === ((1,), (2, 3))
                @test resolve_factorisation(cs4(false), model, fform, (x[i], z)) === ((1,), (2,))
                @test resolve_factorisation(cs4(false), model, fform, (y[i], z)) === ((1,), (2,))
            end
        end

        @testset "Use case #5" begin
            cs = @constraints begin end

            model = FactorGraphModel()

            x = randomvar(model, :x, 11)
            y = randomvar(model, :y, 11)
            z = randomvar(model, :z)

            for i in 1:10
                @test resolve_factorisation(cs, model, fform, (x[i], y[i], z)) === ((1, 2, 3),)
                @test resolve_factorisation(cs, model, fform, (z, x[i], y[i])) === ((1, 2, 3),)
                @test resolve_factorisation(cs, model, fform, (x[i], z, y[i])) === ((1, 2, 3),)
                @test resolve_factorisation(cs, model, fform, (x[i], x[i+1], y[i], y[i+1], z)) === ((1, 2, 3, 4, 5),)
                @test resolve_factorisation(cs, model, fform, (z, x[i], x[i+1], y[i], y[i+1])) === ((1, 2, 3, 4, 5),)
                @test resolve_factorisation(cs, model, fform, (x[i], z, x[i+1], y[i], y[i+1])) === ((1, 2, 3, 4, 5),)
                @test resolve_factorisation(cs, model, fform, (x[i], x[i+1], z, y[i], y[i+1])) === ((1, 2, 3, 4, 5),)
                @test resolve_factorisation(cs, model, fform, (x[i], x[i+1], y[i], z, y[i+1])) === ((1, 2, 3, 4, 5),)
                @test resolve_factorisation(cs, model, fform, (x[i], x[i+1], y[i], y[i+1], z)) === ((1, 2, 3, 4, 5),)
            end
        end

        @testset "Use case #6" begin
            cs = @constraints begin
                q(x) = q(x[begin]) .. q(x[end])
            end

            model = FactorGraphModel()

            x = randomvar(model, :x, 10)

            for i in 1:8
                @test resolve_factorisation(cs, model, fform, (x[i], x[i+1])) === ((1,), (2,))
                @test resolve_factorisation(cs, model, fform, (x[i], x[i+1], x[i+2])) === ((1,), (2,), (3,))
            end
        end

        @testset "Use case #7" begin
            cs = @constraints function cs6(n)
                q(x) = q(x[1:n])q(x[n+1]) .. q(x[end])
            end

            model = FactorGraphModel()

            x = randomvar(model, :x, 10)
            z = randomvar(model, :z)

            for i in 1:10, n in 1:9
                @test resolve_factorisation(cs6(n), model, fform, (x[i], z)) === ((1, 2),)
            end
            @test resolve_factorisation(cs6(5), model, fform, (x[1], x[2])) === ((1, 2),)
            @test resolve_factorisation(cs6(5), model, fform, (x[2], x[1])) === ((1, 2),)
            @test resolve_factorisation(cs6(5), model, fform, (x[5], x[6])) === ((1,), (2,))
            @test resolve_factorisation(cs6(5), model, fform, (x[6], x[5])) === ((1,), (2,))

            @test_throws ReactiveMP.ClusterIntersectionError resolve_factorisation(
                cs6(1),
                model,
                fform,
                (x[1], x[2], z)
            )
            @test_throws ReactiveMP.ClusterIntersectionError resolve_factorisation(
                cs6(1),
                model,
                fform,
                (x[2], x[1], z)
            )
            @test_throws ReactiveMP.ClusterIntersectionError resolve_factorisation(
                cs6(1),
                model,
                fform,
                (z, x[1], x[2])
            )
            @test_throws ReactiveMP.ClusterIntersectionError resolve_factorisation(
                cs6(1),
                model,
                fform,
                (x[1], z, x[2])
            )

            for n in 2:9
                @test resolve_factorisation(cs6(n), model, fform, (x[1], x[2], z)) === ((1, 2, 3),)
            end
        end

        @testset "Use case #8" begin
            @constraints function cs8(flag)
                q(x, y) = q(x[begin], y[begin]) .. q(x[end], y[end])
                q(x, y, t) = q(x, y)q(t)
                q(x, y, r) = q(x, y)q(r)
                if flag
                    q(t, r) = q(t)q(r)
                end
            end

            model = FactorGraphModel()

            y = randomvar(model, :y, 10)
            x = randomvar(model, :x, 10)
            t = randomvar(model, :t)
            r = randomvar(model, :r)

            for i in 1:9
                @test ReactiveMP.resolve_factorisation(cs8(false), model, fform, (y[i], y[i+1], x[i], x[i+1], t, r)) ===
                      ((1, 3), (2, 4), (5, 6))
                @test ReactiveMP.resolve_factorisation(cs8(false), model, fform, (x[i], x[i+1], y[i], y[i+1], t, r)) ===
                      ((1, 3), (2, 4), (5, 6))
                @test ReactiveMP.resolve_factorisation(cs8(false), model, fform, (t, r, x[i], x[i+1], y[i], y[i+1])) ===
                      ((1, 2), (3, 5), (4, 6))
                @test ReactiveMP.resolve_factorisation(cs8(false), model, fform, (t, x[i], x[i+1], y[i], y[i+1], r)) ===
                      ((1, 6), (2, 4), (3, 5))
                @test ReactiveMP.resolve_factorisation(cs8(true), model, fform, (y[i], y[i+1], x[i], x[i+1], t, r)) ===
                      ((1, 3), (2, 4), (5,), (6,))
                @test ReactiveMP.resolve_factorisation(cs8(true), model, fform, (x[i], x[i+1], y[i], y[i+1], t, r)) ===
                      ((1, 3), (2, 4), (5,), (6,))
                @test ReactiveMP.resolve_factorisation(cs8(true), model, fform, (t, r, x[i], x[i+1], y[i], y[i+1])) ===
                      ((1,), (2,), (3, 5), (4, 6))
                @test ReactiveMP.resolve_factorisation(cs8(true), model, fform, (t, x[i], x[i+1], y[i], y[i+1], r)) ===
                      ((1,), (2, 4), (3, 5), (6,))
            end
        end

        @testset "Use case #9" begin
            cs = @constraints begin
                q(x, y) = q(x)q(y)
                q(x, y, t, r) = q(x, y)q(t)q(r)
                q(x, w) = q(x)q(w)
                q(y, w) = q(y)q(w)
                q(x) = q(x[begin:begin+2])q(x[begin+3]) .. q(x[end])
            end

            model = FactorGraphModel()

            x = randomvar(model, :x, 10)
            y = randomvar(model, :y, 10)
            t = randomvar(model, :t, 10)
            r = randomvar(model, :r)

            @test ReactiveMP.resolve_factorisation(cs, model, fform, (x[1], x[2], y[1], t[1])) === ((1, 2), (3,), (4,))
            @test ReactiveMP.resolve_factorisation(cs, model, fform, (x[2], x[3], y[1], t[1])) === ((1, 2), (3,), (4,))
            @test ReactiveMP.resolve_factorisation(cs, model, fform, (x[2], x[3], x[4], y[1], t[1])) ===
                  ((1, 2), (3,), (4,), (5,))
            @test ReactiveMP.resolve_factorisation(cs, model, fform, (x[3], x[4], y[1], t[1])) ===
                  ((1,), (2,), (3,), (4,))
            @test ReactiveMP.resolve_factorisation(cs, model, fform, (x[3], x[4], y[1], t[1], r)) ===
                  ((1,), (2,), (3,), (4,), (5,))
            @test ReactiveMP.resolve_factorisation(cs, model, fform, (x[2], x[3], x[4], y[1], t[1], r)) ===
                  ((1, 2), (3,), (4,), (5,), (6,))
        end

        @testset "Use case #10" begin
            cs = @constraints begin
                q(x, y) = (q(x[begin]) .. q(x[end])) * (q(y[begin]) .. q(y[end]))
                q(x, y, t) = q(x, y)q(t)
                q(x, y, r) = q(x, y)q(r)
            end

            model = FactorGraphModel()

            y = randomvar(model, :y, 10)
            x = randomvar(model, :x, 10)
            t = randomvar(model, :t)
            r = randomvar(model, :r)

            for i in 1:9
                @test ReactiveMP.resolve_factorisation(cs, model, fform, (y[i], y[i+1], x[i], x[i+1], t, r)) ===
                      ((1,), (2,), (3,), (4,), (5, 6))
                @test ReactiveMP.resolve_factorisation(cs, model, fform, (y[i], x[i+1], x[i], y[i+1], t, r)) ===
                      ((1,), (2,), (3,), (4,), (5, 6))
                @test ReactiveMP.resolve_factorisation(cs, model, fform, (x[i], y[i+1], y[i], x[i+1], t, r)) ===
                      ((1,), (2,), (3,), (4,), (5, 6))
                @test ReactiveMP.resolve_factorisation(cs, model, fform, (x[i], x[i+1], y[i], y[i+1], t, r)) ===
                      ((1,), (2,), (3,), (4,), (5, 6))
                @test ReactiveMP.resolve_factorisation(cs, model, fform, (r, y[i], y[i+1], x[i], x[i+1], t)) ===
                      ((1, 6), (2,), (3,), (4,), (5,))
                @test ReactiveMP.resolve_factorisation(cs, model, fform, (t, y[i], y[i+1], x[i], x[i+1], r)) ===
                      ((1, 6), (2,), (3,), (4,), (5,))
                @test ReactiveMP.resolve_factorisation(cs, model, fform, (r, t, y[i], y[i+1], x[i], x[i+1])) ===
                      ((1, 2), (3,), (4,), (5,), (6,))
                @test ReactiveMP.resolve_factorisation(cs, model, fform, (t, r, y[i], y[i+1], x[i], x[i+1])) ===
                      ((1, 2), (3,), (4,), (5,), (6,))
            end
        end

        @testset "Use case #11" begin
            cs = @constraints begin
                q(x, y) = q(y[1])q(x[begin], y[begin+1]) .. q(x[end], y[end])
                q(x, y, t) = q(x, y)q(t)
                q(x, y, r) = q(x, y)q(r)
            end

            model = FactorGraphModel()

            y = randomvar(model, :y, 11)
            x = randomvar(model, :x, 10)
            t = randomvar(model, :t)
            r = randomvar(model, :r)

            @test ReactiveMP.resolve_factorisation(cs, model, fform, (x[1], y[2], x[2], y[3])) === ((1, 2), (3, 4))
            @test ReactiveMP.resolve_factorisation(cs, model, fform, (x[1], y[2], x[2], y[3], y[1])) ===
                  ((1, 2), (3, 4), (5,))
            @test ReactiveMP.resolve_factorisation(cs, model, fform, (y[1], y[2], x[1], x[2], t, r)) ===
                  ((1,), (2, 3), (4,), (5, 6))
            @test ReactiveMP.resolve_factorisation(cs, model, fform, (y[1], y[2], y[3], x[1], x[2], x[3], t, r)) ===
                  ((1,), (2, 4), (3, 5), (6,), (7, 8))
        end

        @testset "Use case #12" begin
            cs = @constraints begin
                q(x, y) = q(x[begin]) * q(x[begin+1:end]) * q(y)
            end

            model = FactorGraphModel()

            y = randomvar(model, :y, 11)
            x = randomvar(model, :x, 10)

            for i in 1:10
                if i > 1 && i < 10
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (x[1], x[i])) === ((1,), (2,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (x[1], x[i], x[i+1])) === ((1,), (2, 3))
                end
                @test ReactiveMP.resolve_factorisation(cs, model, fform, (x[1], x[2], x[3], y[i])) ===
                      ((1,), (2, 3), (4,))
                @test ReactiveMP.resolve_factorisation(cs, model, fform, (x[1], x[2], x[3], x[4], y[i])) ===
                      ((1,), (2, 3, 4), (5,))
            end
        end

        @testset "Use case #13" begin
            cs = @constraints begin
                q(x, y) = q(x)q(y)
            end

            model = FactorGraphModel()

            x = randomvar(model, :x)
            y = randomvar(model, :y)
            tmp = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((y,)), :tmp)
            setanonymous!(tmp, true)

            @test ReactiveMP.resolve_factorisation(cs, model, fform, (x, y)) === ((1,), (2,))
            @test ReactiveMP.resolve_factorisation(cs, model, fform, (x, tmp)) === ((1,), (2,))
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

            model = FactorGraphModel()

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
                tmp1[i] = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((y[i],)), :tmp1)
                tmp2[i] = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((y[i], d[i])), :tmp2)
                tmp3[i] = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((y[i], c[i])), :tmp3)
                tmp4[i] = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((c[i], y[i], d[i])), :tmp4)
                tmp5[i] = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((d[i], y[i], c[i])), :tmp5)
                tmp6[i] = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((d[i], y[i])), :tmp6)
                tmp7[i] = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((c[i], y[i])), :tmp7)

                setanonymous!(tmp1[i], true)
                setanonymous!(tmp2[i], true)
                setanonymous!(tmp3[i], true)
                setanonymous!(tmp4[i], true)
                setanonymous!(tmp5[i], true)
                setanonymous!(tmp6[i], true)
                setanonymous!(tmp7[i], true)
            end

            for i in 1:10
                @test ReactiveMP.resolve_factorisation(cs14(true), model, fform, (x[i], y[i], z)) === ((1,), (2, 3))
                @test ReactiveMP.resolve_factorisation(cs14(true), model, fform, (x[i], tmp1[i], z)) === ((1,), (2, 3))
                @test ReactiveMP.resolve_factorisation(cs14(true), model, fform, (x[i], tmp2[i], z)) === ((1,), (2, 3))
                @test ReactiveMP.resolve_factorisation(cs14(true), model, fform, (x[i], tmp3[i], z)) === ((1,), (2, 3))
                @test ReactiveMP.resolve_factorisation(cs14(true), model, fform, (x[i], tmp4[i], z)) === ((1,), (2, 3))
                @test ReactiveMP.resolve_factorisation(cs14(true), model, fform, (x[i], tmp5[i], z)) === ((1,), (2, 3))
                @test ReactiveMP.resolve_factorisation(cs14(true), model, fform, (x[i], tmp6[i], z)) === ((1,), (2, 3))
                @test ReactiveMP.resolve_factorisation(cs14(true), model, fform, (x[i], tmp7[i], z)) === ((1,), (2, 3))

                @test ReactiveMP.resolve_factorisation(cs14(false), model, fform, (x[i], y[i], z)) === ((1, 3), (2,))
                @test ReactiveMP.resolve_factorisation(cs14(false), model, fform, (x[i], tmp1[i], z)) === ((1, 3), (2,))
                @test ReactiveMP.resolve_factorisation(cs14(false), model, fform, (x[i], tmp2[i], z)) === ((1, 3), (2,))
                @test ReactiveMP.resolve_factorisation(cs14(false), model, fform, (x[i], tmp3[i], z)) === ((1, 3), (2,))
                @test ReactiveMP.resolve_factorisation(cs14(false), model, fform, (x[i], tmp4[i], z)) === ((1, 3), (2,))
                @test ReactiveMP.resolve_factorisation(cs14(false), model, fform, (x[i], tmp5[i], z)) === ((1, 3), (2,))
                @test ReactiveMP.resolve_factorisation(cs14(false), model, fform, (x[i], tmp6[i], z)) === ((1, 3), (2,))
                @test ReactiveMP.resolve_factorisation(cs14(false), model, fform, (x[i], tmp7[i], z)) === ((1, 3), (2,))
            end
        end

        @testset "Use case #15" begin
            # empty and default constraints still should factorize out datavar and constvar
            empty = @constraints begin
                # empty
            end

            # DefaultConstraints are equal to `UnspecifiedConstraints()` for now, but it might change in the future so we test both
            for cs in (empty, UnspecifiedConstraints(), DefaultConstraints)
                let model = FactorGraphModel()
                    d = datavar(model, :d, Float64)
                    c = constvar(model, :c, 1.0)
                    x = randomvar(model, :x)
                    y = randomvar(model, :y)
                    z = randomvar(model, :z)

                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (x, y)) === ((1, 2),)
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (y, x)) === ((1, 2),)
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (d, d)) === ((1,), (2,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (c, c)) === ((1,), (2,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (d, x)) === ((1,), (2,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (d, x, y)) === ((1,), (2, 3))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (x, d, y)) === ((1, 3), (2,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (x, y, d)) === ((1, 2), (3,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (c, x)) === ((1,), (2,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (c, x, y)) === ((1,), (2, 3))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (x, c, y)) === ((1, 3), (2,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (x, y, c)) === ((1, 2), (3,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (c, d)) === ((1,), (2,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (x, y, z)) === ((1, 2, 3),)
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (y, x, z)) === ((1, 2, 3),)
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (z, x, y)) === ((1, 2, 3),)
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (c, x, d)) === ((1,), (2,), (3,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (x, c, d)) === ((1,), (2,), (3,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (x, d, c)) === ((1,), (2,), (3,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (d, x, c)) === ((1,), (2,), (3,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (d, x, d)) === ((1,), (2,), (3,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (d, c, x)) === ((1,), (2,), (3,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (d, d, x)) === ((1,), (2,), (3,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (c, x, d, y)) === ((1,), (2, 4), (3,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (x, c, d, y)) === ((1, 4), (2,), (3,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (x, d, c, y)) === ((1, 4), (2,), (3,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (c, x, y, d)) === ((1,), (2, 3), (4,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (x, c, y, d)) === ((1, 3), (2,), (4,))
                    @test ReactiveMP.resolve_factorisation(cs, model, fform, (x, d, y, c)) === ((1, 3), (2,), (4,))
                end
            end
        end

        @testset "Use case #16" begin
            # Tuple-based variables must be flattened

            model = FactorGraphModel()

            x = randomvar(model, :x)
            y = randomvar(model, :y)
            z = randomvar(model, :z)

            cs1 = @constraints begin 
                q(x, y, z) = q(x)q(y)q(z)
            end

            cs2 = @constraints begin 
                q(x, y, z) = q(x)q(y, z)
            end

            @test ReactiveMP.resolve_factorisation(cs1, model, fform, (x, (y, z))) === ((1,), (2,), (3,))
            @test ReactiveMP.resolve_factorisation(cs2, model, fform, (x, (y, z))) === ((1,), (2, 3,))
            @test ReactiveMP.resolve_factorisation(cs1, model, fform, ((x, y), z)) === ((1,), (2,), (3,))
            @test ReactiveMP.resolve_factorisation(cs2, model, fform, ((x, y), z)) === ((1,), (2, 3,))
        end

        ## Warning testing below

        # Variable does not exist
        @testset "Warning case #1" begin
            model = FactorGraphModel()

            cs_with_warn = @constraints [warn = true] begin
                q(x) = q(x[begin]) .. q(x[end])
            end

            cs_without_warn = @constraints [warn = false] begin
                q(x) = q(x[begin]) .. q(x[end])
            end

            @test_logs (:warn, r".*q(.*).*has no random variable") activate!(cs_with_warn, model)
            @test_logs min_level = Logging.Warn activate!(cs_without_warn, model)
        end

        @testset "Warning case #2" begin
            # datavar in factorisation constraint
            model = FactorGraphModel()

            x = datavar(model, :x, Float64)

            cs_with_warn = @constraints [warn = true] begin
                q(x) = q(x[begin]) .. q(x[end])
            end

            cs_without_warn = @constraints [warn = false] begin
                q(x) = q(x[begin]) .. q(x[end])
            end

            @test_logs (:warn, r".*q(.*).*is not a random variable") activate!(cs_with_warn, model)
            @test_logs min_level = Logging.Warn activate!(cs_without_warn, model)
        end

        @testset "Warning case #3" begin
            # constvar in factorisation constraint
            model = FactorGraphModel()

            x = constvar(model, :x, 1.0)

            cs_with_warn = @constraints [warn = true] begin
                q(x) = q(x[begin]) .. q(x[end])
            end

            cs_without_warn = @constraints [warn = false] begin
                q(x) = q(x[begin]) .. q(x[end])
            end

            @test_logs (:warn, r".*q(.*).*is not a random variable") activate!(cs_with_warn, model)
            @test_logs min_level = Logging.Warn activate!(cs_without_warn, model)
        end

        ## Error testing below

        @testset "Error case #1" begin
            # Names are not unique
            @test_throws ErrorException @constraints begin
                q(x, y) = q(x[begin], x[begin]) .. q(x[end], x[end])q(y)
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

            model = FactorGraphModel()

            x = randomvar(model, :x)
            y = randomvar(model, :y)

            z = randomvar(model, :z)
            d = datavar(model, :d, Float64)
            c = constvar(model, :c, 1)

            tmp1 = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((x, y)), :tmp1)
            tmp2 = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((x, y, d)), :tmp2)
            tmp3 = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((x, y, c)), :tmp3)
            tmp4 = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((d, x, y)), :tmp4)
            tmp5 = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((x, c, y)), :tmp5)
            tmp6 = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((x, d, y)), :tmp6)
            tmp7 = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((c, x, y)), :tmp7)
            tmp8 = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((c, x, y, d)), :tmp8)
            tmp9 = randomvar(model, ReactiveMP.randomvar_options_set_proxy_variables((d, x, y, c)), :tmp9)

            setanonymous!(tmp1, true)
            setanonymous!(tmp2, true)
            setanonymous!(tmp3, true)
            setanonymous!(tmp4, true)
            setanonymous!(tmp5, true)
            setanonymous!(tmp6, true)
            setanonymous!(tmp7, true)
            setanonymous!(tmp8, true)
            setanonymous!(tmp9, true)

            @test_throws ErrorException resolve_factorisation(cs, model, fform, (z, tmp1))
            @test_throws ErrorException resolve_factorisation(cs, model, fform, (z, tmp2))
            @test_throws ErrorException resolve_factorisation(cs, model, fform, (z, tmp3))
            @test_throws ErrorException resolve_factorisation(cs, model, fform, (z, tmp4))
            @test_throws ErrorException resolve_factorisation(cs, model, fform, (z, tmp5))
            @test_throws ErrorException resolve_factorisation(cs, model, fform, (z, tmp6))
            @test_throws ErrorException resolve_factorisation(cs, model, fform, (z, tmp7))
            @test_throws ErrorException resolve_factorisation(cs, model, fform, (z, tmp8))
            @test_throws ErrorException resolve_factorisation(cs, model, fform, (z, tmp9))
        end

        @testset "Error case #5" begin
            # undefined variables
            model = FactorGraphModel()
            other = FactorGraphModel()

            x = randomvar(other, :x)
            y = randomvar(other, :y)

            cs = @constraints begin
                q(x, y) = q(x)q(y)
            end

            @test_throws ErrorException ReactiveMP.resolve_factorisation(cs, model, :Normal, (x, y))
        end
    end
end

end
