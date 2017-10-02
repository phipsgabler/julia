# This file is a part of Julia. License is MIT: https://julialang.org/license

# Uniform random generation

## State machinery

rand(rng::AbstractRNG, X) = rand(rng, State(rng, X))
rand(rng::AbstractRNG=GLOBAL_RNG, ::Type{X}=Float64) where {X} = rand(rng, State(rng, X))

rand(X) = rand(GLOBAL_RNG, X)
rand(::Type{X}) where X = rand(GLOBAL_RNG, X)

rand!(A::AbstractArray{T}, X) where {T} = rand!(GLOBAL_RNG, A, X)
rand!(A::AbstractArray{T}, ::Type{X}=T) where {T,X} = rand!(GLOBAL_RNG, A, X)

rand!(rng::AbstractRNG, A::AbstractArray{T}, X) where {T} = rand!(rng, A, State(rng, X))
rand!(rng::AbstractRNG, A::AbstractArray{T}, ::Type{X}=T) where {T,X} = rand!(rng, A, State(rng, X))

### arrays of random numbers

function rand!(rng::AbstractRNG, A::AbstractArray{T}, st::State) where T
    for i in eachindex(A)
        @inbounds A[i] = rand(rng, st)
    end
    A
end

typemap(::Type{T}) where {T} = T
typemap(::AbstractArray{T}) where {T} = T
typemap(::AbstractSet{T}) where {T} = T
typemap(::Associative{K,V}) where {K,V} = Pair{K,V}
typemap(::AbstractString) = Char
typemap(::FloatInterval{T}) where {T} = T

rand(r::AbstractRNG, dims::Dims)       = rand(r, Float64, dims)
rand(                dims::Dims)       = rand(GLOBAL_RNG, dims)
rand(r::AbstractRNG, dims::Integer...) = rand(r, Dims(dims))
rand(                dims::Integer...) = rand(Dims(dims))

rand(r::AbstractRNG, X, dims::Dims)  = rand!(r, Array{typemap(X)}(dims), X)
rand(                X, dims::Dims)  = rand(GLOBAL_RNG, X, dims)

rand(r::AbstractRNG, X, d::Integer, dims::Integer...) = rand(r, X, Dims((d, dims...)))
rand(                X, d::Integer, dims::Integer...) = rand(X, Dims((d, dims...)))
# note: the above methods would trigger an ambiguity warning if d was not separated out:
# rand(r, ()) would match both this method and rand(r, dims::Dims)
# moreover, a call like rand(r, NotImplementedType()) would be an infinite loop

rand(r::AbstractRNG, ::Type{X}, dims::Dims) where {X} = rand!(r, Array{typemap(X)}(dims), X)
rand(                ::Type{X}, dims::Dims) where {X} = rand(GLOBAL_RNG, X, dims)

rand(r::AbstractRNG, ::Type{X}, d::Integer, dims::Integer...) where {X} = rand(r, X, Dims((d, dims...)))
rand(                ::Type{X}, d::Integer, dims::Integer...) where {X} = rand(X, Dims((d, dims...)))

## from types: rand(::Type, [dims...])

### random floats

State(rng::AbstractRNG, ::Type{T}) where {T<:AbstractFloat} = State(rng, CloseOpen(T))

# generic random generation function which can be used by RNG implementors
# it is not defined as a fallback rand method as this could create ambiguities

rand_generic(r::AbstractRNG, ::CloseOpen{Float16}) =
    Float16(reinterpret(Float32,
                        (rand_ui10_raw(r) % UInt32 << 13) & 0x007fe000 | 0x3f800000) - 1)

rand_generic(r::AbstractRNG, ::CloseOpen{Float32}) =
    reinterpret(Float32, rand_ui23_raw(r) % UInt32 & 0x007fffff | 0x3f800000) - 1

rand_generic(r::AbstractRNG, ::Close1Open2_64) =
    reinterpret(Float64, 0x3ff0000000000000 | rand(r, UInt64) & 0x000fffffffffffff)

rand_generic(r::AbstractRNG, ::CloseOpen_64) = rand(r, Close1Open2()) - 1.0

#### BigFloat

const bits_in_Limb = sizeof(Limb) << 3
const Limb_high_bit = one(Limb) << (bits_in_Limb-1)

struct StateBigFloat{I<:FloatInterval{BigFloat}} <: State
    prec::Int
    nlimbs::Int
    limbs::Vector{Limb}
    shift::UInt

    function StateBigFloat{I}(prec::Int) where I<:FloatInterval{BigFloat}
        nlimbs = (prec-1) ÷ bits_in_Limb + 1
        limbs = Vector{Limb}(nlimbs)
        shift = nlimbs * bits_in_Limb - prec
        new(prec, nlimbs, limbs, shift)
    end
end


State(I::FloatInterval{BigFloat}) = StateBigFloat{typeof(I)}(precision(BigFloat))

function _rand(rng::AbstractRNG, st::StateBigFloat)
    z = BigFloat()
    limbs = st.limbs
    rand!(rng, limbs)
    @inbounds begin
        limbs[1] <<= st.shift
        randbool = iszero(limbs[end] & Limb_high_bit)
        limbs[end] |= Limb_high_bit
    end
    z.sign = 1
    Base.@gc_preserve limbs unsafe_copy!(z.d, pointer(limbs), st.nlimbs)
    (z, randbool)
end

function _rand(rng::AbstractRNG, st::StateBigFloat, ::Close1Open2{BigFloat})
    z = _rand(rng, st)[1]
    z.exp = 1
    z
end

function _rand(rng::AbstractRNG, st::StateBigFloat, ::CloseOpen{BigFloat})
    z, randbool = _rand(rng, st)
    z.exp = 0
    randbool &&
        ccall((:mpfr_sub_d, :libmpfr), Int32,
              (Ref{BigFloat}, Ref{BigFloat}, Cdouble, Int32),
              z, z, 0.5, Base.MPFR.ROUNDING_MODE[])
    z
end

# alternative, with 1 bit less of precision
# TODO: make an API for requesting full or not-full precision
function _rand(rng::AbstractRNG, st::StateBigFloat{CloseOpen{BigFloat}}, ::Void)
    z = _rand(rng, st, Close1Open2(BigFloat))
    ccall((:mpfr_sub_ui, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Culong, Int32),
          z, z, 1, Base.MPFR.ROUNDING_MODE[])
    z
end

rand(rng::AbstractRNG, st::StateBigFloat{T}) where {T<:FloatInterval{BigFloat}} =
    _rand(rng, st, T())

### random integers

rand_ui10_raw(r::AbstractRNG) = rand(r, UInt16)
rand_ui23_raw(r::AbstractRNG) = rand(r, UInt32)

rand_ui52_raw(r::AbstractRNG) = reinterpret(UInt64, rand(r, Close1Open2()))
rand_ui52(r::AbstractRNG) = rand_ui52_raw(r) & 0x000fffffffffffff

### random complex numbers

rand(r::AbstractRNG, ::StateType{Complex{T}}) where {T<:Real} =
    complex(rand(r, T), rand(r, T))

### random characters

# returns a random valid Unicode scalar value (i.e. 0 - 0xd7ff, 0xe000 - # 0x10ffff)
function rand(r::AbstractRNG, ::StateType{Char})
    c = rand(r, 0x00000000:0x0010f7ff)
    (c < 0xd800) ? Char(c) : Char(c+0x800)
end

## Generate random integer within a range

### BitInteger

# remainder function according to Knuth, where rem_knuth(a, 0) = a
rem_knuth(a::UInt, b::UInt) = a % (b + (b == 0)) + a * (b == 0)
rem_knuth(a::T, b::T) where {T<:Unsigned} = b != 0 ? a % b : a

# maximum multiple of k <= 2^bits(T) decremented by one,
# that is 0xFFFF...FFFF if k = typemax(T) - typemin(T) with intentional underflow
# see http://stackoverflow.com/questions/29182036/integer-arithmetic-add-1-to-uint-max-and-divide-by-n-without-overflow
maxmultiple(k::T) where {T<:Unsigned} =
    (div(typemax(T) - k + oneunit(k), k + (k == 0))*k + k - oneunit(k))::T

# maximum multiple of k within 1:2^32 or 1:2^64 decremented by one, depending on size
maxmultiplemix(k::UInt64) = k >> 32 != 0 ?
    maxmultiple(k) :
    (div(0x0000000100000000, k + (k == 0))*k - oneunit(k))::UInt64

struct StateRangeInt{T<:Integer,U<:Unsigned} <: State
    a::T   # first element of the range
    k::U   # range length or zero for full range
    u::U   # rejection threshold
end

# generators with 32, 128 bits entropy
StateRangeInt(a::T, k::U) where {T,U<:Union{UInt32,UInt128}} =
    StateRangeInt{T,U}(a, k, maxmultiple(k))

# mixed 32/64 bits entropy generator
StateRangeInt(a::T, k::UInt64) where {T} =
    StateRangeInt{T,UInt64}(a, k, maxmultiplemix(k))

function State(::AbstractRNG, r::UnitRange{T}) where T<:Unsigned
    isempty(r) && throw(ArgumentError("range must be non-empty"))
    StateRangeInt(first(r), last(r) - first(r) + oneunit(T))
end

for (T, U) in [(UInt8, UInt32), (UInt16, UInt32),
               (Int8, UInt32), (Int16, UInt32), (Int32, UInt32),
               (Int64, UInt64), (Int128, UInt128), (Bool, UInt32)]

    @eval State(::AbstractRNG, r::UnitRange{$T}) = begin
        isempty(r) && throw(ArgumentError("range must be non-empty"))
        # overflow ok:
        StateRangeInt(first(r), convert($U, unsigned(last(r) - first(r)) + one($U)))
    end
end

# this function uses 32 bit entropy for small ranges of length <= typemax(UInt32) + 1
# StateRangeInt is responsible for providing the right value of k
function rand(rng::AbstractRNG, st::StateRangeInt{T,UInt64}) where T<:Union{UInt64,Int64}
    local x::UInt64
    if (st.k - 1) >> 32 == 0
        x = rand(rng, UInt32)
        while x > st.u
            x = rand(rng, UInt32)
        end
    else
        x = rand(rng, UInt64)
        while x > st.u
            x = rand(rng, UInt64)
        end
    end
    return reinterpret(T, reinterpret(UInt64, st.a) + rem_knuth(x, st.k))
end

function rand(rng::AbstractRNG, st::StateRangeInt{T,U}) where {T<:Integer,U<:Unsigned}
    x = rand(rng, U)
    while x > st.u
        x = rand(rng, U)
    end
    (unsigned(st.a) + rem_knuth(x, st.k)) % T
end

### BigInt

struct StateBigInt <: State
    a::BigInt         # first
    m::BigInt         # range length - 1
    nlimbs::Int       # number of limbs in generated BigInt's (z ∈ [0, m])
    nlimbsmax::Int    # max number of limbs for z+a
    mask::Limb        # applied to the highest limb
end

function State(::AbstractRNG, r::UnitRange{BigInt})
    m = last(r) - first(r)
    m < 0 && throw(ArgumentError("range must be non-empty"))
    nd = ndigits(m, 2)
    nlimbs, highbits = divrem(nd, 8*sizeof(Limb))
    highbits > 0 && (nlimbs += 1)
    mask = highbits == 0 ? ~zero(Limb) : one(Limb)<<highbits - one(Limb)
    nlimbsmax = max(nlimbs, abs(last(r).size), abs(first(r).size))
    return StateBigInt(first(r), m, nlimbs, nlimbsmax, mask)
end

function rand(rng::AbstractRNG, st::StateBigInt)
    x = MPZ.realloc2(st.nlimbsmax*8*sizeof(Limb))
    limbs = unsafe_wrap(Array, x.d, st.nlimbs)
    while true
        rand!(rng, limbs)
        @inbounds limbs[end] &= st.mask
        MPZ.mpn_cmp(x, st.m, st.nlimbs) <= 0 && break
    end
    # adjust x.size (normally done by mpz_limbs_finish, in GMP version >= 6)
    x.size = st.nlimbs
    while x.size > 0
        @inbounds limbs[x.size] != 0 && break
        x.size -= 1
    end
    MPZ.add!(x, st.a)
end

## random values from AbstractArray

State(rng::AbstractRNG, r::AbstractArray) = StateSimple(r, State(rng, 1:length(r)))

rand(rng::AbstractRNG, st::StateSimple{<:AbstractArray,<:State}) =
    @inbounds return st[][rand(rng, st.state)]


## random values from Dict, Set, IntSet

function rand(rng::AbstractRNG, st::StateTrivial{<:Dict})
    isempty(st[]) && throw(ArgumentError("collection must be non-empty"))
    rst = State(rng, 1:length(st[].slots))
    while true
        i = rand(rng, rst)
        Base.isslotfilled(st[], i) && @inbounds return (st[].keys[i] => st[].vals[i])
    end
end

rand(rng::AbstractRNG, st::StateTrivial{<:Set}) = rand(rng, st[].dict).first

function rand(rng::AbstractRNG, st::StateTrivial{IntSet})
    isempty(st[]) && throw(ArgumentError("collection must be non-empty"))
    # st[] can be empty while st[].bits is not, so we cannot rely on the
    # length check in State below
    rst = State(rng, 1:length(st[].bits))
    while true
        n = rand(rng, rst)
        @inbounds b = st[].bits[n]
        b && return n
    end
end

### generic containers

# avoid linear complexity for repeated calls
State(rng::AbstractRNG, s::Union{Associative,AbstractSet}) = State(rng, collect(s))
State(::AbstractRNG, s::Union{Set,Dict,IntSet}) = StateTrivial(s)

# when generating only one element, avoid the call to collect

function nth(iter, n::Integer)::eltype(iter)
    for (i, x) in enumerate(iter)
        i == n && return x
    end
end

rand(rng::AbstractRNG, t::Union{Associative,AbstractSet}) =
    nth(t, rand(rng, 1:length(t)))

rand(rng::AbstractRNG, t::Union{Set,IntSet,Dict}) = rand(rng, State(rng, t))

## random characters from a string

# we use collect(str), which is most of the time more efficient than specialized methods
# (except maybe for very small arrays)
State(rng::AbstractRNG, str::AbstractString) = State(rng, collect(str))

# when generating only one char from a string, the specialized method below
# is usually more efficient
isvalid_unsafe(s::String, i) = !Base.is_valid_continuation(Base.@gc_preserve s unsafe_load(pointer(s), i))
isvalid_unsafe(s::AbstractString, i) = isvalid(s, i)
_endof(s::String) = sizeof(s)
_endof(s::AbstractString) = endof(s)

function rand(rng::AbstractRNG, str::AbstractString)::Char
    st = State(rng, 1:_endof(str))
    while true
        pos = rand(rng, st)
        isvalid_unsafe(str, pos) && return str[pos]
    end
end
