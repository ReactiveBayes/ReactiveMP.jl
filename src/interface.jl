using Rocket

export AbstractInterface, InterfaceIn, InterfaceOut

abstract type AbstractInterface end

define_joint!(interface, observable)       = set!(interface.joint, observable)
define_sum_product!(interface, observable) = set!(interface.sum_product, observable)

sum_product(interface) = interface.sum_product
joint(interface)       = interface.joint

# struct Interface
#     sum_product_msg :: LazyObservable{AbstractMessage}
#     partner_msg     :: LazyObservable{AbstractMessage}
#
#     Interface(name::String) = new(name, lazy(AbstractMessage), lazy(AbstractMessage))
# end

struct InterfaceIn <: AbstractInterface
    name :: String

    sum_product :: LazyObservable{AbstractMessage}
    joint       :: LazyObservable{AbstractMessage}

    InterfaceIn(name::String) = begin
        sum_product = LazyObservable{AbstractMessage}()
        joint       = LazyObservable{AbstractMessage}()
        return new(name, sum_product, joint)
    end
end

struct InterfaceOut <: AbstractInterface
    name :: String

    sum_product :: LazyObservable{AbstractMessage}
    joint       :: LazyObservable{AbstractMessage}

    InterfaceOut(name::String) = begin
        sum_product = LazyObservable{AbstractMessage}()
        joint       = LazyObservable{AbstractMessage}()
        return new(name, sum_product, joint)
    end
end
