module ReactiveMPFlattenTupleTest

using Test
using ReactiveMP: flatten_tuple

@testset "Flatten Tuple" begin

    @testset "Operation" begin

        @test flatten_tuple(1)                  == (1,)
        @test flatten_tuple("str")              == ("str",)
        @test flatten_tuple(5.0)                == (5.0,)

        @test flatten_tuple((1,))               == (1,)
        @test flatten_tuple(("str",))           == ("str",)
        @test flatten_tuple((5.0,))             == (5.0,)

        @test flatten_tuple(())                 == ()

        @test flatten_tuple((1, 2, 3))          == (1, 2, 3)
        @test flatten_tuple((3.0, 2.0, 1.0))    == (3.0, 2.0, 1.0)
        @test flatten_tuple(("test", 1.0, 5))   == ("test", 1.0, 5)

        @test flatten_tuple(((((1),),),))       == (1,)
        @test flatten_tuple(((((1, 2.0),),),))  == (1, 2.0)
        @test flatten_tuple(((((1, 2.0, "str"), (0.5, "srt")),()),(4,)))  == (1, 2.0, "str", 0.5, "srt", 4)

    end

end
end