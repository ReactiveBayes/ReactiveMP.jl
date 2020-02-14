export Interface, sum_product_msg, partner_msg
export define_sum_product_msg!, define_partner_msg!

using Rocket

import Base: show

struct Interface
    name                :: String
    sum_product_message :: LazyObservable{AbstractMessage}
    partner_message     :: LazyObservable{AbstractMessage}

    Interface(name::String) = new(name, lazy(AbstractMessage), lazy(AbstractMessage))
end

sum_product_message(interface::Interface) = interface.sum_product_message
partner_message(interface::Interface)     = interface.partner_message

define_sum_product_message!(interface::Interface, source) = set!(interface.sum_product_message, source)
define_partner_message!(interface::Interface,     source) = set!(interface.partner_message, source)

Base.show(io::IO, interface::Interface) = print(io, "Interface($(interface.name))")


