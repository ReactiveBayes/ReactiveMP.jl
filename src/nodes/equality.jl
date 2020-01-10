export EqualityIOONode
export equality_ioo

@CreateMapOperator(MultipleIOO, Tuple{StochasticMessage{Normal{Float64}}, StochasticMessage{Normal{Float64}}}, StochasticMessage{Normal{Float64}}, (d::Tuple{StochasticMessage{Normal{Float64}}, StochasticMessage{Normal{Float64}}}) -> multiply(d[1], d[2]))

struct EqualityIOONode <: AbstractFactorNode
    name :: String

    in1  :: InterfaceIn{StochasticMessage{Normal{Float64}}, StochasticMessage{Normal{Float64}}}
    out1 :: InterfaceOut{StochasticMessage{Normal{Float64}}, StochasticMessage{Normal{Float64}}}
    out2 :: InterfaceOut{StochasticMessage{Normal{Float64}}, StochasticMessage{Normal{Float64}}}

    EqualityIOONode(name::String) = begin
        in1  = InterfaceIn{StochasticMessage{Normal{Float64}}, StochasticMessage{Normal{Float64}}}("[$name]: in1InterfaceIn")
        out1 = InterfaceOut{StochasticMessage{Normal{Float64}}, StochasticMessage{Normal{Float64}}}("[$name]: out1InterfaceOut")
        out2 = InterfaceOut{StochasticMessage{Normal{Float64}}, StochasticMessage{Normal{Float64}}}("[$name]: out2InterfaceOut")

        define_sum_product!(in1, combineLatest(out1.joint_message, out2.joint_message) |> MultipleIOOMapOperator())

        define_sum_product!(out1, combineLatest(in1.joint_message, out2.joint_message) |> MultipleIOOMapOperator())

        define_sum_product!(out2, combineLatest(out1.joint_message, in1.joint_message) |> MultipleIOOMapOperator())

        return new(name, in1, out1, out2)
    end
end

equality_ioo(name::String) = EqualityIOONode(name)
