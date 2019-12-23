using Rx

export AbstractInterface, InterfaceIn, InterfaceOut

abstract type AbstractInterface end

function define_joint!(interface::AbstractInterface, observable)
    define!(interface.joint_message, observable)
end

function define_sum_product!(interface::AbstractInterface, observable)
    define!(interface.sum_product_message, observable)
end

function sum_product(interface::AbstractInterface)
    return interface.sum_product_message
end

struct InterfaceIn <: AbstractInterface
    name :: String

    joint_message       :: LazyObservable{AbstractMessage}
    sum_product_message :: LazyObservable{AbstractMessage}

    InterfaceIn(name::String) = begin
        joint_message       = LazyObservable{AbstractMessage}("[$name]: jointMessage")
        sum_product_message = LazyObservable{AbstractMessage}("[$name]: sumProductMessage")

        return new(name, joint_message, sum_product_message)
    end
end

struct InterfaceOut <: AbstractInterface
    name :: String

    joint_message       :: LazyObservable{AbstractMessage}
    sum_product_message :: LazyObservable{AbstractMessage}

    InterfaceOut(name::String) = begin
        joint_message       = LazyObservable{AbstractMessage}("[$name]: jointMessage")
        sum_product_message = LazyObservable{AbstractMessage}("[$name]: sumProductMessage")

        return new(name, joint_message, sum_product_message)
    end
end
