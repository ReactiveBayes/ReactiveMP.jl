using Tullio

# --------------- Rules for 2 interfaces (q_out PointMass) ---------------
@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass{<:AbstractVector}, q_a::PointMass{<:AbstractArray{T, 2}}, meta::Any) where {T} = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    out = eloga' * probvec(q_out)
    out .= exp.(out)
    return Categorical(normalize!(out, 1); check_args = false)
    softmax!(out)
    return Categorical(out; check_args = false)
end

# --------------- Rules for 2 interfaces (q_out PointMass, q_a DirichletCollection) ---------------
@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass{<:AbstractVector}, q_a::DirichletCollection, meta::Any) = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    out = eloga' * probvec(q_out)
    softmax!(out)
    return Categorical(out; check_args = false)
end

# --------------- Rules for 3 interfaces (q_out PointMass, q_a PointMass) ---------------
@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass{<:AbstractVector}, q_a::PointMass{<:AbstractArray{T, 3}}, m_T1::DiscreteNonParametric, meta::Any) where {T} = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio out[i, j] := eloga[a, i, j] * probvec(q_out)[a]
    softmax!(out)
    msg = out * probvec(m_T1)
    return Categorical(normalize!(msg, 1); check_args = false)
end

@rule DiscreteTransition(:T1, Marginalisation) (q_out::PointMass{<:AbstractVector}, m_in::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 3}}, meta::Any) where {T} = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio out[i, j] := eloga[a, i, j] * probvec(q_out)[a]
    softmax!(out)
    msg = out' * probvec(m_in)
    return Categorical(normalize!(msg, 1); check_args = false)
end

# --------------- Rules for 3 interfaces (q_out PointMass, q_a DirichletCollection) ---------------
@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass{<:AbstractVector}, q_a::DirichletCollection, m_T1::DiscreteNonParametric, meta::Any) = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio out[i, j] := eloga[a, i, j] * probvec(q_out)[a]
    softmax!(out)
    msg = out * probvec(m_T1)
    return Categorical(normalize!(msg, 1); check_args = false)
end

@rule DiscreteTransition(:T1, Marginalisation) (q_out::PointMass{<:AbstractVector}, m_in::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio out[i, j] := eloga[a, i, j] * probvec(q_out)[a]
    softmax!(out)
    msg = out' * probvec(m_in)
    return Categorical(normalize!(msg, 1); check_args = false)
end

# --------------- Rules for 4 interfaces (q_out PointMass, q_a PointMass) ---------------
@rule DiscreteTransition(:in, Marginalisation) (
    q_out::PointMass{<:AbstractVector}, q_a::PointMass{<:AbstractArray{T, 4}}, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, meta::Any
) where {T} = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio out[i, j, k] := eloga[a, i, j, k] * probvec(q_out)[a]
    softmax!(out)
    @tullio msg[i] := out[i, j, k] * probvec(m_T1)[j] * probvec(m_T2)[k]
    return Categorical(normalize!(msg, 1); check_args = false)
end

@rule DiscreteTransition(:T1, Marginalisation) (
    q_out::PointMass{<:AbstractVector}, m_in::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 4}}, m_T2::DiscreteNonParametric, meta::Any
) where {T} = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio out[i, j, k] := eloga[a, i, j, k] * probvec(q_out)[a]
    softmax!(out)
    @tullio msg[j] := out[i, j, k] * probvec(m_in)[i] * probvec(m_T2)[k]
    return Categorical(normalize!(msg, 1); check_args = false)
end

@rule DiscreteTransition(:T2, Marginalisation) (
    q_out::PointMass{<:AbstractVector}, m_in::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 4}}, m_T1::DiscreteNonParametric, meta::Any
) where {T} = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio out[i, j, k] := eloga[a, i, j, k] * probvec(q_out)[a]
    softmax!(out)
    @tullio msg[k] := out[i, j, k] * probvec(m_in)[i] * probvec(m_T1)[j]
    return Categorical(normalize!(msg, 1); check_args = false)
end

# --------------- Rules for 4 interfaces (q_out PointMass, q_a DirichletCollection) ---------------
@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass{<:AbstractVector}, q_a::DirichletCollection, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, meta::Any) =
    begin
        eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
        @tullio out[i, j, k] := eloga[a, i, j, k] * probvec(q_out)[a]
        softmax!(out)
        @tullio msg[i] := out[i, j, k] * probvec(m_T1)[j] * probvec(m_T2)[k]
        return Categorical(normalize!(msg, 1); check_args = false)
    end

@rule DiscreteTransition(:T1, Marginalisation) (q_out::PointMass{<:AbstractVector}, m_in::DiscreteNonParametric, q_a::DirichletCollection, m_T2::DiscreteNonParametric, meta::Any) =
    begin
        eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
        @tullio out[i, j, k] := eloga[a, i, j, k] * probvec(q_out)[a]
        softmax!(out)
        @tullio msg[j] := out[i, j, k] * probvec(m_in)[i] * probvec(m_T2)[k]
        return Categorical(normalize!(msg, 1); check_args = false)
    end

@rule DiscreteTransition(:T2, Marginalisation) (q_out::PointMass{<:AbstractVector}, m_in::DiscreteNonParametric, q_a::DirichletCollection, m_T1::DiscreteNonParametric, meta::Any) =
    begin
        eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
        @tullio out[i, j, k] := eloga[a, i, j, k] * probvec(q_out)[a]
        softmax!(out)
        @tullio msg[k] := out[i, j, k] * probvec(m_in)[i] * probvec(m_T1)[j]
        return Categorical(normalize!(msg, 1); check_args = false)
    end

# --------------- Rules for 5 interfaces (q_out PointMass, q_a PointMass) ---------------
@rule DiscreteTransition(:in, Marginalisation) (
    q_out::PointMass{<:AbstractVector}, q_a::PointMass{<:AbstractArray{T, 5}}, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, meta::Any
) where {T} = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio out[i, j, k, l] := eloga[a, i, j, k, l] * probvec(q_out)[a]
    softmax!(out)
    @tullio msg[i] := out[i, j, k, l] * probvec(m_T1)[j] * probvec(m_T2)[k] * probvec(m_T3)[l]
    return Categorical(normalize!(msg, 1); check_args = false)
end

@rule DiscreteTransition(:T1, Marginalisation) (
    q_out::PointMass{<:AbstractVector}, m_in::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 5}}, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, meta::Any
) where {T} = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio out[i, j, k, l] := eloga[a, i, j, k, l] * probvec(q_out)[a]
    softmax!(out)
    @tullio msg[j] := out[i, j, k, l] * probvec(m_in)[i] * probvec(m_T2)[k] * probvec(m_T3)[l]
    return Categorical(normalize!(msg, 1); check_args = false)
end

@rule DiscreteTransition(:T2, Marginalisation) (
    q_out::PointMass{<:AbstractVector}, m_in::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 5}}, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, meta::Any
) where {T} = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio out[i, j, k, l] := eloga[a, i, j, k, l] * probvec(q_out)[a]
    softmax!(out)
    @tullio msg[k] := out[i, j, k, l] * probvec(m_in)[i] * probvec(m_T1)[j] * probvec(m_T2)[l]
    return Categorical(normalize!(msg, 1); check_args = false)
end

@rule DiscreteTransition(:T3, Marginalisation) (
    q_out::PointMass{<:AbstractVector}, m_in::DiscreteNonParametric, q_a::PointMass{<:AbstractArray{T, 5}}, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, meta::Any
) where {T} = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio out[i, j, k, l] := eloga[a, i, j, k, l] * probvec(q_out)[a]
    softmax!(out)
    @tullio msg[l] := out[i, j, k, l] * probvec(m_in)[i] * probvec(m_T1)[j] * probvec(m_T2)[k]
    return Categorical(normalize!(msg, 1); check_args = false)
end

# --------------- Rules for 5 interfaces (q_out PointMass, q_a DirichletCollection) ---------------
@rule DiscreteTransition(:in, Marginalisation) (
    q_out::PointMass{<:AbstractVector}, q_a::DirichletCollection, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, meta::Any
) = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio out[i, j, k, l] := eloga[a, i, j, k, l] * probvec(q_out)[a]
    softmax!(out)
    @tullio msg[i] := out[i, j, k, l] * probvec(m_T1)[j] * probvec(m_T2)[k] * probvec(m_T3)[l]
    return Categorical(normalize!(msg, 1); check_args = false)
end

@rule DiscreteTransition(:T1, Marginalisation) (
    q_out::PointMass{<:AbstractVector}, m_in::DiscreteNonParametric, q_a::DirichletCollection, m_T2::DiscreteNonParametric, m_T3::DiscreteNonParametric, meta::Any
) = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio out[i, j, k, l] := eloga[a, i, j, k, l] * probvec(q_out)[a]
    softmax!(out)
    @tullio msg[j] := out[i, j, k, l] * probvec(m_in)[i] * probvec(m_T2)[k] * probvec(m_T3)[l]
    return Categorical(normalize!(msg, 1); check_args = false)
end

@rule DiscreteTransition(:T2, Marginalisation) (
    q_out::PointMass{<:AbstractVector}, m_in::DiscreteNonParametric, q_a::DirichletCollection, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, meta::Any
) = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio out[i, j, k, l] := eloga[a, i, j, k, l] * probvec(q_out)[a]
    softmax!(out)
    @tullio msg[k] := out[i, j, k, l] * probvec(m_in)[i] * probvec(m_T1)[j] * probvec(m_T2)[l]
    return Categorical(normalize!(msg, 1); check_args = false)
end

@rule DiscreteTransition(:T3, Marginalisation) (
    q_out::PointMass{<:AbstractVector}, m_in::DiscreteNonParametric, q_a::DirichletCollection, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, meta::Any
) = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio out[i, j, k, l] := eloga[a, i, j, k, l] * probvec(q_out)[a]
    softmax!(out)
    @tullio msg[l] := out[i, j, k, l] * probvec(m_in)[i] * probvec(m_T1)[j] * probvec(m_T2)[k]
    return Categorical(normalize!(msg, 1); check_args = false)
end
