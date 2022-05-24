module ReactiveMPMathTest

using Test
using ReactiveMP
using Random

@testset "Math" begin
    @testset "tiny/huge" begin
        @test typeof(tiny) === TinyNumber
        @test typeof(huge) === HugeNumber

        @test convert(Float32, tiny) == 1.0f-6
        @test convert(Float64, tiny) == 1e-12
        @test convert(BigFloat, tiny) == big"1e-24"

        @test Float32(tiny) == 1.0f-6
        @test Float64(tiny) == 1e-12
        @test BigFloat(tiny) == big"1e-24"

        @test convert(Float32, huge) == 1.0f+6
        @test convert(Float64, huge) == 1e+12
        @test convert(BigFloat, huge) == big"1e+24"

        @test Float32(huge) == 1.0f+6
        @test Float64(huge) == 1e+12
        @test BigFloat(huge) == big"1e+24"

        @test @inferred clamp(1.0f0, tiny, huge) == 1.0f0
        @test @inferred clamp(0.0f0, tiny, huge) == 1.0f-6
        @test @inferred clamp(1.0f13, tiny, huge) == 1.0f+6

        @test @inferred clamp(1.0, tiny, huge) == 1.0
        @test @inferred clamp(0.0, tiny, huge) == 1e-12
        @test @inferred clamp(1e13, tiny, huge) == 1e12

        @test @inferred clamp(big"1.0", tiny, huge) == big"1.0"
        @test @inferred clamp(big"0.0", tiny, huge) == big"1e-24"
        @test @inferred clamp(big"1e25", tiny, huge) == big"1e+24"

        for a in (1, 1.0, 0, 0.0, 1.0f0, 0.0f0, Int32(0), Int32(1), big"1", big"1.0", big"0", big"0.0")
            T = typeof(a)
            for v in [tiny, huge]
                V = typeof(v)

                for op in [+, -, *, /, >, >=, <, <=]
                    @test @inferred op(a, v) == op(a, convert(promote_type(T, V), v))
                    @test @inferred op(v, a) == op(convert(promote_type(T, V), v), a)

                    @test @inferred op(a, v) == op(a, v)
                    @test @inferred op(v, a) == op(v, a)
                end

                @test v <= (@inferred clamp(a, v, Inf)) <= Inf
                @test zero(a) <= (@inferred clamp(a, zero(a), v)) <= v

                for size in [3, 5]
                    for array in [fill(a, (size,)), fill(a, (size, size))]
                        for op in [+, -, *, /, >, >=, <, <=]
                            @test @inferred op.(array, v) == op.(array, convert(promote_type(T, V), v))
                            @test @inferred op.(v, array) == op.(convert(promote_type(T, V), v), array)

                            @test @inferred op.(array, v) == op.(array, v)
                            @test @inferred op.(v, array) == op.(v, array)
                        end

                        @test @inferred clamp.(array, v, Inf) == clamp.(array, v, Inf)
                        @test @inferred clamp.(array, zero(array), v) == clamp.(array, zero(array), v)
                    end
                end
            end
        end
    end
end

end
