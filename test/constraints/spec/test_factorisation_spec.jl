module ReactiveMPFactorisationSpecTest 

using Test
using ReactiveMP 

import ReactiveMP: FactorisationSpecEntryIndex
import ReactiveMP: FactorisationSpecEntryExact, FactorisationSpecEntryIndexed, FactorisationSpecEntryRanged, FactorisationSpecEntrySplitRanged
import ReactiveMP: SplittedRange
import ReactiveMP: FactorisationSpecEntry, FactorisationSpec, FactorisationSpecNode
import ReactiveMP: indextype

@testset "show" begin 
    @test repr(FactorisationSpecEntry(:z, nothing)) == "z"
    @test repr(FactorisationSpecEntry(:x, nothing)) == "x"
    @test repr(FactorisationSpecEntry(:z, 1)) == "z[1]"
    @test repr(FactorisationSpecEntry(:x, 3)) == "x[3]"
    @test repr(FactorisationSpecEntry(:z, 1:4)) == "z[1:4]"
    @test repr(FactorisationSpecEntry(:x, 6:9)) == "x[6:9]"
    @test repr(FactorisationSpecEntry(:z, SplittedRange(1:4))) == "z[1]..z[4]"
    @test repr(FactorisationSpecEntry(:x, SplittedRange(6:9))) == "x[6]..x[9]"

    @test repr(FactorisationSpec(( FactorisationSpecEntry(:z, 1), FactorisationSpecEntry(:s, 1) ))) == "q(z[1], s[1])"
    @test repr(FactorisationSpec(( FactorisationSpecEntry(:z, 3), FactorisationSpecEntry(:s, 1) ))) == "q(z[3], s[1])"
    @test repr(FactorisationSpec(( FactorisationSpecEntry(:z, 3), FactorisationSpecEntry(:s, nothing) ))) == "q(z[3], s)"
    @test repr(FactorisationSpec(( FactorisationSpecEntry(:z, 1:2), FactorisationSpecEntry(:s, 1:2) ))) == "q(z[1:2], s[1:2])"
    @test repr(FactorisationSpec(( FactorisationSpecEntry(:z, 1:2), FactorisationSpecEntry(:s, 1:2) ))) == "q(z[1:2], s[1:2])"
    @test repr(FactorisationSpec(( FactorisationSpecEntry(:z, SplittedRange(1:2)), FactorisationSpecEntry(:s, nothing) ))) == "q(z[1]..z[2], s)"
end

@testset "indextype" begin 
    
    @test indextype(FactorisationSpecEntry(:z, nothing)) === FactorisationSpecEntryExact()
    @test indextype(FactorisationSpecEntry(:z, 1)) === FactorisationSpecEntryIndexed()
    @test indextype(FactorisationSpecEntry(:z, 1:2)) === FactorisationSpecEntryRanged()
    @test indextype(FactorisationSpecEntry(:z, SplittedRange(1:2))) === FactorisationSpecEntrySplitRanged()

end

@testset "merge!" begin 

    @test_throws ErrorException merge!(FactorisationSpecEntry(:z, nothing), FactorisationSpecEntry(:s, nothing))
    @test merge!(FactorisationSpecEntry(:s, nothing), FactorisationSpecEntry(:s, nothing)) == FactorisationSpecEntry(:s, nothing)
    @test_throws ErrorException merge!(FactorisationSpecEntry(:z, nothing), FactorisationSpecEntry(:z, 1))
    @test_throws ErrorException merge!(FactorisationSpecEntry(:z, 1), FactorisationSpecEntry(:z, nothing))
    @test_throws ErrorException merge!(FactorisationSpecEntry(:z, 1), FactorisationSpecEntry(:s, 2))
    @test merge!(FactorisationSpecEntry(:z, 1), FactorisationSpecEntry(:z, 3)) == FactorisationSpecEntry(:z, SplittedRange(1:3))
    @test_throws AssertionError merge!(FactorisationSpecEntry(:z, 3), FactorisationSpecEntry(:z, 1)) == FactorisationSpecEntry(:z, 1:1)
    @test merge!(FactorisationSpecEntry(:z, 1),FactorisationSpecEntry(:z, SplittedRange(2:4)),) == FactorisationSpecEntry(:z, SplittedRange(1:4))
    @test merge!(FactorisationSpecEntry(:z, SplittedRange(1:5)),FactorisationSpecEntry(:z, 8),) == FactorisationSpecEntry(:z, SplittedRange(1:8))
    @test merge!(FactorisationSpecEntry(:z, SplittedRange(1:5)),FactorisationSpecEntry(:z, SplittedRange(6:10)),) == FactorisationSpecEntry(:z, SplittedRange(1:10))
    @test_throws AssertionError merge!(FactorisationSpecEntry(:z, 10),FactorisationSpecEntry(:z, SplittedRange(2:4)),) == FactorisationSpecEntry(:z, SplittedRange(1:4))
    @test_throws AssertionError merge!(FactorisationSpecEntry(:z, SplittedRange(1:10)),FactorisationSpecEntry(:z, 8),) == FactorisationSpecEntry(:z, SplittedRange(1:8))
    @test_throws AssertionError merge!(FactorisationSpecEntry(:z, SplittedRange(1:10)),FactorisationSpecEntry(:z, SplittedRange(6:10)),) == FactorisationSpecEntry(:z, SplittedRange(1:10))

    @test merge!(FactorisationSpec(( FactorisationSpecEntry(:z, 1), FactorisationSpecEntry(:s, 1) )), FactorisationSpec(( FactorisationSpecEntry(:z, 4), FactorisationSpecEntry(:s, 4) ))) == FactorisationSpec(( FactorisationSpecEntry(:z, SplittedRange(1:4)), FactorisationSpecEntry(:s, SplittedRange(1:4)) ))
    @test_throws ErrorException merge!(FactorisationSpec(( FactorisationSpecEntry(:s, 1), FactorisationSpecEntry(:z, 1) )), FactorisationSpec(( FactorisationSpecEntry(:z, 4), FactorisationSpecEntry(:s, 4) ))) == FactorisationSpec(( FactorisationSpecEntry(:s, SplittedRange(1:4)), FactorisationSpecEntry(:z, SplittedRange(1:4)) ))
    @test merge!(FactorisationSpec(( FactorisationSpecEntry(:s, nothing), FactorisationSpecEntry(:z, 1) )), FactorisationSpec(( FactorisationSpecEntry(:s, nothing), FactorisationSpecEntry(:z, 4) ))) == FactorisationSpec(( FactorisationSpecEntry(:s, nothing), FactorisationSpecEntry(:z, SplittedRange(1:4)) ))

    @test_throws ErrorException merge!(FactorisationSpec(( FactorisationSpecEntry(:z, 1), )), FactorisationSpec(( FactorisationSpecEntry(:z, 4), FactorisationSpecEntry(:s, nothing) ))) 
    @test_throws ErrorException merge!(FactorisationSpec(( FactorisationSpecEntry(:x, 1), FactorisationSpecEntry(:s, 1) )), FactorisationSpec(( FactorisationSpecEntry(:z, 4), FactorisationSpecEntry(:s, 4) )))
end

@testset "hash" begin 
    @test hash(SplittedRange(1:2)) === hash(SplittedRange(1:2))
    @test hash(SplittedRange(1:2)) != hash(1:2)
    @test hash(FactorisationSpecEntry(:z, 1:2)) === hash(FactorisationSpecEntry(:z, 1:2))
    @test hash(FactorisationSpecEntry(:z, 1:3)) != hash(FactorisationSpecEntry(:z, 1:2))
    @test hash(FactorisationSpecEntry(:s, 1:2)) != hash(FactorisationSpecEntry(:z, 1:2))
    @test hash(FactorisationSpecEntry(:z, SplittedRange(1:2))) != hash(FactorisationSpecEntry(:z, 1:2))

    @test hash(FactorisationSpec(( FactorisationSpecEntry(:z, nothing), FactorisationSpecEntry(:s, nothing) ))) == hash(FactorisationSpec(( FactorisationSpecEntry(:z, nothing), FactorisationSpecEntry(:s, nothing) )))
    @test hash(FactorisationSpec(( FactorisationSpecEntry(:z, SplittedRange(1:2)), FactorisationSpecEntry(:s, SplittedRange(1:2)) ))) == hash(FactorisationSpec(( FactorisationSpecEntry(:z, SplittedRange(1:2)), FactorisationSpecEntry(:s, SplittedRange(1:2)) )))
    @test hash(FactorisationSpec(( FactorisationSpecEntry(:z, SplittedRange(1:2)), FactorisationSpecEntry(:s, SplittedRange(1:2)) ))) != hash(FactorisationSpec(( FactorisationSpecEntry(:z, SplittedRange(1:3)), FactorisationSpecEntry(:s, SplittedRange(1:3)) )))
    @test hash(FactorisationSpec(( FactorisationSpecEntry(:z, nothing), FactorisationSpecEntry(:x, nothing) ))) != hash(FactorisationSpec(( FactorisationSpecEntry(:z, nothing), FactorisationSpecEntry(:s, nothing) )))

    testdict = Dict{FactorisationSpec, Any}()

    entry1 = FactorisationSpec(( FactorisationSpecEntry(:z, nothing), FactorisationSpecEntry(:s, nothing) ))
    entry2 = FactorisationSpec(( FactorisationSpecEntry(:z, nothing), FactorisationSpecEntry(:s, nothing) ))
    testdict[entry1] = 2

    @test testdict[entry2] === 2

    entry3 = merge!(FactorisationSpec(( FactorisationSpecEntry(:z, 1), FactorisationSpecEntry(:s, 1) )), FactorisationSpec(( FactorisationSpecEntry(:z, 4), FactorisationSpecEntry(:s, 4) ))) 
    entry4 = FactorisationSpec(( FactorisationSpecEntry(:z, SplittedRange(1:4)), FactorisationSpecEntry(:s, SplittedRange(1:4)) ))

    testdict[entry3] = 3
    @test testdict[entry4] === 3
end

## 

end