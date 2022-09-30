module ReactiveMPProdAnalyticalTest

using Test
using ReactiveMP

import ReactiveMP: prod, NoAnalyticalProdException

@testset "ProdAnalytical" begin
    @test prod(ProdAnalytical(), missing, missing) === missing

    struct ProdAnalyticalTestStructLeft end
    struct ProdAnalyticalTestStructRight end

    ReactiveMP.prod(::ProdAnalytical, ::ProdAnalyticalTestStructLeft, ::ProdAnalyticalTestStructRight) = 1

    @test prod(ProdAnalytical(), ProdAnalyticalTestStructLeft(), ProdAnalyticalTestStructRight()) === 1
    @test prod(ProdAnalytical(), ProdAnalyticalTestStructLeft(), missing) === ProdAnalyticalTestStructLeft()
    @test prod(ProdAnalytical(), missing, ProdAnalyticalTestStructRight()) === ProdAnalyticalTestStructRight()

    @test_throws NoAnalyticalProdException prod(ProdAnalytical(), ProdAnalyticalTestStructLeft(), ProdAnalyticalTestStructLeft())
    @test_throws NoAnalyticalProdException prod(ProdAnalytical(), ProdAnalyticalTestStructRight(), ProdAnalyticalTestStructRight())

    errmsg = sprint(showerror, NoAnalyticalProdException(ProdAnalyticalTestStructLeft(), ProdAnalyticalTestStructLeft()))

    @test occursin("No analytical rule available", errmsg)
    @test occursin("Possible fix", errmsg)
end

end
