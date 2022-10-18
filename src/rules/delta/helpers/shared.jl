

# ported from ForneyLab.jl

function collectStatistics(msgs::Vararg{Any})
    stats = []
    for msg in msgs
        (msg === nothing) && continue # Skip unreported messages
        push!(stats, mean_cov(msg))
    end

    ms = [stat[1] for stat in stats]
    Vs = [stat[2] for stat in stats]
    return (ms, Vs) # Return tuple with vectors for means and covariances
end

function collectStatistics(msg::NormalDistributionsFamily)
    return mean_cov(msg)
end
