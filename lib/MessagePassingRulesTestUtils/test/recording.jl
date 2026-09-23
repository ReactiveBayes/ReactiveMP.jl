@testmodule Recording begin
    using Test

    mutable struct RecordingTestSet <: Test.AbstractTestSet
        description::String
        results::Vector{Any}
    end
    RecordingTestSet(description; kwargs...) = RecordingTestSet(description, Any[])
    Test.record(set::RecordingTestSet, result) = (push!(set.results, result); result)
    Test.finish(set::RecordingTestSet) = set

    passes(set) = count(r -> r isa Test.Pass, set.results)
    failures(set) = filter(r -> r isa Test.Fail || r isa Test.Error, set.results)
    failure_text(set) = join((sprint(show, f) for f in failures(set)), "\n")

    # Runs `f` with every check recorded here instead of in the enclosing test set.
    function recorded(f)
        return Test.@testset RecordingTestSet "recorded" begin
            f()
        end
    end
end
