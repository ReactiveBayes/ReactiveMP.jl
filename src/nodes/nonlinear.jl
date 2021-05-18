export NonLinear, NonLinearMetadata

struct NonLinearMetadata
    f :: Function
    df :: Function
    fi :: Function
    dfi :: Function
end

struct NonLinear end

@node NonLinear Deterministic [ out, x ]