module ReactiveMPScoreCountingTest

using Test
using ReactiveMP

import ReactiveMP: CountingReal, ImproperReal

@testset "CountingReal" begin
    r = CountingReal(0.0, 0)
    @test float(r) ≈ 0.0
    @test float(r + 1) ≈ 1.0
    @test float(1 + r) ≈ 1.0
    @test float(r - 1) ≈ -1.0
    @test float(1 - r) ≈ 1.0
    @test float(r - 1 + ImproperReal()) ≈ Inf
    @test float(1 - r + ImproperReal()) ≈ Inf
    @test float(r - 1 + ImproperReal() - ImproperReal()) ≈ -1.0
    @test float(1 - r + ImproperReal() - ImproperReal()) ≈ 1.0
end

end
