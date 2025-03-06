using BenchmarkTools
using ReactiveMP
import ReactiveMP: Marginal, Contingency, Categorical, Bernoulli, PointMass, Message, marginalrule, normalize!
using ReactiveMP.BayesBase
using ReactiveMP.ExponentialFamily

function add_discrete_transition_marginals_benchmarks(SUITE)
    SUITE["Marginals"] = BenchmarkGroup(["Marginals", "DiscreteTransition"])
    for n_categories in 4:5
        for categorical_size in [3, 5, 10, 20]
            discrete_transition_fast_marginal_rule(SUITE["Marginals"], n_categories, categorical_size)
            discrete_transition_generic_marginal_rule(SUITE["Marginals"], n_categories, categorical_size)
        end
    end
end

# Benchmark for the fast implementation (joint marginal over all categoricals)
function discrete_transition_fast_marginal_rule(SUITE, n_categories, cat_size)
    # Create messages for all interfaces except 'a'
    incoming_messages = ntuple(_ -> ReactiveMP.Message(Categorical(normalize!(rand(cat_size), 1)), false, false, nothing), n_categories - 1)

    # Create message names
    incoming_messages_name = [:in]
    for i in 1:(n_categories - 2)
        push!(incoming_messages_name, Symbol("t$i"))
    end
    incoming_messages_name = Tuple(incoming_messages_name)

    # Create marginal for 'a'
    incoming_marginal = (ReactiveMP.Marginal(DirichletCollection(rand(ntuple(_ -> cat_size, n_categories)...)), false, false, nothing),)

    # Set up the benchmark
    SUITE["Fast Implementation"]["$(n_categories) categories, $(cat_size) size"] = @benchmarkable marginalrule(
        DiscreteTransition, Val(:out), Val($incoming_messages_name), $incoming_messages, Val((:a,)), $incoming_marginal, nothing, nothing
    )
end

# Benchmark for the generic implementation
function discrete_transition_generic_marginal_rule(SUITE, n_categories, cat_size)
    # Create messages for some interfaces
    n_messages = div(n_categories, 2)  # Use half of the interfaces as messages
    incoming_messages = ntuple(_ -> ReactiveMP.Message(Categorical(normalize!(rand(cat_size), 1)), false, false, nothing), n_messages)

    # Create message names
    incoming_messages_name = []
    for i in 1:(n_messages)
        if i == 1
            push!(incoming_messages_name, :out)
        elseif i == 2
            push!(incoming_messages_name, :in)
        else
            push!(incoming_messages_name, Symbol("t$(i-2)"))
        end
    end
    incoming_messages_name = Tuple(incoming_messages_name)
    out_marginal_name = Symbol(join(incoming_messages_name, "_"))

    # Create marginals for the remaining interfaces including 'a'
    n_marginals = n_categories - n_messages + 1  # +1 for 'a'
    incoming_marginals = ntuple(i -> if i == 1
        ReactiveMP.Marginal(DirichletCollection(rand(ntuple(_ -> cat_size, n_categories)...)), false, false, nothing)
    else
        ReactiveMP.Marginal(Categorical(normalize!(rand(cat_size), 1)), false, false, nothing)
    end, n_marginals)

    # Create marginal names
    incoming_marginals_name = [:a]
    for i in n_messages:(n_categories - 1)
        push!(incoming_marginals_name, Symbol("t$(i-1)"))
    end
    incoming_marginals_name = Tuple(incoming_marginals_name)

    # Set up the benchmark
    SUITE["Generic Implementation"]["$(n_categories) categories, $(cat_size) size"] = @benchmarkable marginalrule(
        DiscreteTransition, Val($out_marginal_name), Val($incoming_messages_name), $incoming_messages, Val($incoming_marginals_name), $incoming_marginals, nothing, nothing
    )
end
