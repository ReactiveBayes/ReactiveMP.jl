@testitem "Helpers" begin
    using ReactiveMP

    import ReactiveMP: SkipIndexIterator, skipindex
    import ReactiveMP: CountingReal
    import ReactiveMP: FunctionalIndex

    @testset "SkipIndexIterator" begin
        s = skipindex(1:3, 2)
        @test typeof(s) <: SkipIndexIterator
        @test collect(s) == [1, 3]
        @test collect(skipindex(s, 1)) == [3]
    end
end
