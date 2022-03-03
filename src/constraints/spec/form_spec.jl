
struct FormConstraintsSpecification{T, A, K}
    args   :: A
    kwargs :: K
end

FormConstraintsSpecification(::Type{T}, args::A, kwargs::K) where { T, A, K } = FormConstraintsSpecification{T, A, K}(args, kwargs)

function Base.show(io::IO, spec::FormConstraintsSpecification{T}) where T
    print(io, "$T")
    print(io, "(")

    if length(spec.args) >= 1
        join(io, spec.args, ", ")
    end

    if length(spec.kwargs) >= 1
        print(io, "; ")
        join(io, map(p -> string(first(p), " = ", last(p)), collect(pairs(spec.kwargs))), ", ")
    end

    print(io, ")")
end