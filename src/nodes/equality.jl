export EqualityIOONode

using Rocket

Rocket.@GenerateCombineLatest(2, "equalityMessage", AbstractMessage, true, t -> multiply(t[1], t[2]))

struct EqualityIOONode <: AbstractDeterministicNode
    name :: String
    in1  :: Interface
    out1 :: Interface
    out2 :: Interface

    EqualityIOONode(name::String) = begin
        in1  = Interface("[$name]: in1")
        out1 = Interface("[$name]: out1")
        out2 = Interface("[$name]: out2")

        # define_sum_product!(in1, combineLatest(joint(out1), joint(out2)) |> MultipleIOOMapOperator())
        define_sum_product_message!(in1,  equalityMessage(partner_message(out1), partner_message(out2)) |> share_replay(1, mode = SYNCHRONOUS_SUBJECT_MODE))

        define_sum_product_message!(out1, equalityMessage(partner_message(in1), partner_message(out2)))

        define_sum_product_message!(out2, equalityMessage(partner_message(out1), partner_message(in1)))

        return new(name, in1, out1, out2)
    end
end
