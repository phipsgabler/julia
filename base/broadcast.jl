# This file is a part of Julia. License is MIT: https://julialang.org/license

module Broadcast

using Base.Cartesian
using Base: Bottom, Indices, OneTo, linearindices, tail, to_shape,
            _msk_end, unsafe_bitgetindex, bitcache_chunks, bitcache_size, dumpbitcache,
            nullable_returntype, null_safe_op, hasvalue, isoperator
import Base: broadcast, broadcast!
export broadcast_getindex, broadcast_setindex!, dotview, @__dot__

# Note: `similar` and `indices` will be overridden below, thus you need to use
# Base.similar and Base.indices when you want the Base versions.

"""
    result = Broadcast.Result{ContainerType}()
    result = Broadcast.Result{ContainerType,ElType}(inds::Indices)

Create an object that specifies the type and (optionally) indices
of the result (output) of a broadcasting operation.

Using a dedicated type for this information makes it possible to support
variants of [`broadcast`](@ref) that accept `result` as an argument;
it prevents an ambiguity of intent that would otherwise arise because
both types and indices-tuples are among the supported *input*
arguments to `broadcast`. For example, `parse.(Int, ("1", "2"))` is
equivalent to `broadcast(parse, Int, ("1", "2"))`, and as a consequence
it would would be ambiguous if result-type and output-indices information
were passed as positional arguments to `broadcast`.

You can extract `inds` with `indices(result)`.
"""
struct Result{ContainerType,ElType,I<:Union{Void,Indices}}
    indices::I
end
Result{ContainerType}() where ContainerType =
    Result{ContainerType,Void,Void}(nothing)
Result{ContainerType,ElType}(inds::Indices) where {ContainerType,ElType} =
    Result{ContainerType,ElType,typeof(inds)}(inds)
indices(r::Result) = r.indices
Base.indices(r::Result) = indices(r)
Base.eltype(r::Result{ContainerType,ElType}) where {ContainerType,ElType} = ElType

# An AbstractArray type that "loses" in precedence comparisons to all other AbstractArrays
abstract type BottomArray{T,N} <: AbstractArray{T,N} end

### User-extensible methods (see the Interfaces chapter of the manual) ###
## Computing the result (output) type
"""
    Broadcast.rule(::Type{<:MyContainer}) = MyContainer

Declare that objects of type `MyContainer` have a customized broadcast implementation.
If you define this method, you are responsible for defining the following method:

    Base.similar(f, r::Broadcast.Result{MyContainer}, As...) = ...

where `f` is the function you're broadcasting, `r` is a [`Broadcast.Result`](@ref)
indicating the eltype and indices of the output container, and `As...` contains
the input arguments to `broadcast`.
"""
rule(::Type{Bottom}) = Bottom
rule(::Type{<:Ptr})  = Bottom   # Ptrs act like scalars, not like Ref
rule(::Type{T}) where T = Bottom   # "scalar" behavior
rule(::Type{<:Nullable}) = Nullable
rule(::Type{<:Tuple}) = Tuple
rule(::Type{<:Ref}) = BottomArray
rule(::Type{<:AbstractArray}) = BottomArray

"""
    Broadcast.rule(::Type{S}, ::Type{T}) where {S,T} = U

Indicate how to resolve different broadcast `rule`s. For example,

    Broadcast.rule(::Type{Primary}, ::Type{Secondary}) = Primary

would indicate that `Primary` has precedence over `Secondary`.
In most cases you do not have to define both orders, and overloading just the
first argument is less likely to cause method ambiguities.

In general the inputs should not use subtyping, e.g., use `Type{Primary}` rather than
`Type{<:Primary}`, because `Union{}` (which is a subtype of every type) plays an important
role in `rule`. As a consequence, if you define binary `rule` methods, when defining unary
`rule` methods you should typically discard any container type parameters.
"""
function rule(::Type{S}, ::Type{T}) where {S,T}
    # Precedence rules. We avoid ambiguities by having an order in which these tests are performed.
    S == T && return T
    S == Union{} && return T
    T == Union{} && return S
    S<:AbstractArray && T==BottomArray && return S
    S==BottomArray && T<:AbstractArray && return T
    S<:AbstractArray && T<:AbstractArray && return BottomArray # in cases of ambiguity default to BottomArray
    S<:AbstractArray && return S
    T<:AbstractArray && return T
    S == Tuple && return S
    T == Tuple && return T
    S == Nullable && return S
    T == Nullable && return T
    Bottom
end

## Computing the result's indices
indices() = ()
indices(::Type{T}) where T = ()
indices(A) = indices(resulttype(A), A)
indices(::Type{Bottom}, A) = ()
indices(::Type{Any}, A) = ()
indices(::Type{Nullable}, A) = ()
indices(::Type{Tuple}, A) = (OneTo(length(A)),)
indices(::Type{BottomArray}, A::Ref) = ()
indices(::Type{<:AbstractArray}, A) = Base.indices(A)

combine_indices(A, B...) = broadcast_shape(indices(A), combine_indices(B...))
combine_indices(A) = indices(A)

## Allocating the output container
"""
    dest = similar(f, r::Broadcast.Result{ContainerType}, As...)

Allocate an output object `dest`, of the type signaled by `ContainerType`,  for [`broadcast`](@ref).
`f` is the broadcast operations, and `As...` are the arguments supplied to `broadcast`.
See [`Broadcast.Result`](@ref) and [`Broadcast.rule`](@ref).
"""
Base.similar(f, r::Result{BottomArray}, As...) = similar(Array{eltype(r)}, indices(r))

## Broadcasting utilities ##
# special cases
broadcast(f, x::Number...) = f(x...)
broadcast(f, t::NTuple{N,Any}, ts::Vararg{NTuple{N,Any}}) where {N} = map(f, t, ts...)
broadcast!(::typeof(identity), x::AbstractArray{T,N}, y::AbstractArray{S,N}) where {T,S,N} =
    Base.indices(x) == Base.indices(y) ? copy!(x, y) : _broadcast!(identity, x, y)

# special cases for "X .= ..." (broadcast!) assignments
broadcast!(::typeof(identity), X::AbstractArray, x::Number) = fill!(X, x)
broadcast!(f, X::AbstractArray, x::Number...) = (@inbounds for I in eachindex(X); X[I] = f(x...); end; X)

# logic for deciding the resulting container type
resulttype(c) = resultt(rule(typeof(c)))
resulttype(c1, c2) = resultt(resulttype(c1), resulttype(c2))
resulttype(c1, c2, cs...) = resultt(resulttype(c1), resulttype(c2, cs...))

resultt(::Type{T}) where T = T
resultt(::Type{T}, ::Type{T}) where T = T
# Test both orders so users typically only have to specialize the first argument
resultt(::Type{S}, ::Type{T}) where {S,T} = _resultt(rule(S, T), rule(T, S))
_resultt(::Type{S}, ::Type{T}) where {S<:AbstractArray,T<:AbstractArray} = rule(S, T)
_resultt(::Type{S}, ::Type{T}) where {S,T} = promote_type(S, T)

# shape (i.e., tuple-of-indices) inputs
broadcast_shape(shape::Tuple) = shape
broadcast_shape(shape::Tuple, shape1::Tuple, shapes::Tuple...) = broadcast_shape(_bcs(shape, shape1), shapes...)
# _bcs consolidates two shapes into a single output shape
_bcs(::Tuple{}, ::Tuple{}) = ()
_bcs(::Tuple{}, newshape::Tuple) = (newshape[1], _bcs((), tail(newshape))...)
_bcs(shape::Tuple, ::Tuple{}) = (shape[1], _bcs(tail(shape), ())...)
function _bcs(shape::Tuple, newshape::Tuple)
    return (_bcs1(shape[1], newshape[1]), _bcs(tail(shape), tail(newshape))...)
end
# _bcs1 handles the logic for a single dimension
_bcs1(a::Integer, b::Integer) = a == 1 ? b : (b == 1 ? a : (a == b ? a : throw(DimensionMismatch("arrays could not be broadcast to a common size"))))
_bcs1(a::Integer, b) = a == 1 ? b : (first(b) == 1 && last(b) == a ? b : throw(DimensionMismatch("arrays could not be broadcast to a common size")))
_bcs1(a, b::Integer) = _bcs1(b, a)
_bcs1(a, b) = _bcsm(b, a) ? b : (_bcsm(a, b) ? a : throw(DimensionMismatch("arrays could not be broadcast to a common size")))
# _bcsm tests whether the second index is consistent with the first
_bcsm(a, b) = a == b || length(b) == 1
_bcsm(a, b::Number) = b == 1
_bcsm(a::Number, b::Number) = a == b || b == 1

## Check that all arguments are broadcast compatible with shape
# comparing one input against a shape
check_broadcast_shape(shp) = nothing
check_broadcast_shape(shp, ::Tuple{}) = nothing
check_broadcast_shape(::Tuple{}, ::Tuple{}) = nothing
check_broadcast_shape(::Tuple{}, Ashp::Tuple) = throw(DimensionMismatch("cannot broadcast array to have fewer dimensions"))
function check_broadcast_shape(shp, Ashp::Tuple)
    _bcsm(shp[1], Ashp[1]) || throw(DimensionMismatch("array could not be broadcast to match destination"))
    check_broadcast_shape(tail(shp), tail(Ashp))
end
check_broadcast_indices(shp, A) = check_broadcast_shape(shp, indices(A))
# comparing many inputs
@inline function check_broadcast_indices(shp, A, As...)
    check_broadcast_indices(shp, A)
    check_broadcast_indices(shp, As...)
end

## Indexing manipulations

# newindex(I, keep, Idefault) replaces a CartesianIndex `I` with something that
# is appropriate for a particular broadcast array/scalar. `keep` is a
# NTuple{N,Bool}, where keep[d] == true means that one should preserve
# I[d]; if false, replace it with Idefault[d].
# If dot-broadcasting were already defined, this would be `ifelse.(keep, I, Idefault)`.
@inline newindex(I::CartesianIndex, keep, Idefault) = CartesianIndex(_newindex(I.I, keep, Idefault))
@inline _newindex(I, keep, Idefault) =
    (ifelse(keep[1], I[1], Idefault[1]), _newindex(tail(I), tail(keep), tail(Idefault))...)
@inline _newindex(I, keep::Tuple{}, Idefault) = ()  # truncate if keep is shorter than I

# newindexer(shape, A) generates `keep` and `Idefault` (for use by
# `newindex` above) for a particular array `A`, given the
# broadcast indices `shape`
# `keep` is equivalent to map(==, indices(A), shape) (but see #17126)
@inline newindexer(shape, A) = shapeindexer(shape, indices(A))
@inline shapeindexer(shape, indsA::Tuple{}) = (), ()
@inline function shapeindexer(shape, indsA::Tuple)
    ind1 = indsA[1]
    keep, Idefault = shapeindexer(tail(shape), tail(indsA))
    (shape[1] == ind1, keep...), (first(ind1), Idefault...)
end

# Equivalent to map(x->newindexer(shape, x), As) (but see #17126)
map_newindexer(shape, ::Tuple{}) = (), ()
@inline function map_newindexer(shape, As)
    A1 = As[1]
    keeps, Idefaults = map_newindexer(shape, tail(As))
    keep, Idefault = newindexer(shape, A1)
    (keep, keeps...), (Idefault, Idefaults...)
end
@inline function map_newindexer(shape, A, Bs)
    keeps, Idefaults = map_newindexer(shape, Bs)
    keep, Idefault = newindexer(shape, A)
    (keep, keeps...), (Idefault, Idefaults...)
end

Base.@propagate_inbounds _broadcast_getindex(::Type{T}, I) where T = T
Base.@propagate_inbounds _broadcast_getindex(A, I) = _broadcast_getindex(resulttype(A), A, I)
Base.@propagate_inbounds _broadcast_getindex(::Type{BottomArray}, A::Ref, I) = A[]
Base.@propagate_inbounds _broadcast_getindex(::Union{Type{Bottom},Type{Any},Type{Nullable}}, A, I) = A
Base.@propagate_inbounds _broadcast_getindex(::Type, A, I) = A[I]

## Broadcasting core
# nargs encodes the number of As arguments (which matches the number
# of keeps). The first two type parameters are to ensure specialization.
@generated function _broadcast!(f, B::AbstractArray, keeps::K, Idefaults::ID, A::AT, Bs::BT, ::Val{N}, iter) where {K,ID,AT,BT,N}
    nargs = N + 1
    quote
        $(Expr(:meta, :inline))
        # destructure the keeps and As tuples
        A_1 = A
        @nexprs $N i->(A_{i+1} = Bs[i])
        @nexprs $nargs i->(keep_i = keeps[i])
        @nexprs $nargs i->(Idefault_i = Idefaults[i])
        @simd for I in iter
            # reverse-broadcast the indices
            @nexprs $nargs i->(I_i = newindex(I, keep_i, Idefault_i))
            # extract array values
            @nexprs $nargs i->(@inbounds val_i = _broadcast_getindex(A_i, I_i))
            # call the function and store the result
            result = @ncall $nargs f val
            @inbounds B[I] = result
        end
    end
end

# For BitArray outputs, we cache the result in a "small" Vector{Bool},
# and then copy in chunks into the output
@generated function _broadcast!(f, B::BitArray, keeps::K, Idefaults::ID, A::AT, Bs::BT, ::Val{N}, iter) where {K,ID,AT,BT,N}
    nargs = N + 1
    quote
        $(Expr(:meta, :inline))
        # destructure the keeps and As tuples
        A_1 = A
        @nexprs $N i->(A_{i+1} = Bs[i])
        @nexprs $nargs i->(keep_i = keeps[i])
        @nexprs $nargs i->(Idefault_i = Idefaults[i])
        C = Vector{Bool}(bitcache_size)
        Bc = B.chunks
        ind = 1
        cind = 1
        @simd for I in iter
            # reverse-broadcast the indices
            @nexprs $nargs i->(I_i = newindex(I, keep_i, Idefault_i))
            # extract array values
            @nexprs $nargs i->(@inbounds val_i = _broadcast_getindex(A_i, I_i))
            # call the function and store the result
            @inbounds C[ind] = @ncall $nargs f val
            ind += 1
            if ind > bitcache_size
                dumpbitcache(Bc, cind, C)
                cind += bitcache_chunks
                ind = 1
            end
        end
        if ind > 1
            @inbounds C[ind:bitcache_size] = false
            dumpbitcache(Bc, cind, C)
        end
    end
end

"""
    broadcast!(f, dest, As...)

Like [`broadcast`](@ref), but store the result of
`broadcast(f, As...)` in the `dest` array.
Note that `dest` is only used to store the result, and does not supply
arguments to `f` unless it is also listed in the `As`,
as in `broadcast!(f, A, A, B)` to perform `A[:] = broadcast(f, A, B)`.
"""
@inline broadcast!(f, C::AbstractArray, A, Bs::Vararg{Any,N}) where {N} =
    _broadcast!(f, C, A, Bs...)

# This indirection allows size-dependent implementations (e.g., see the copying `identity`
# specialization above)
@inline function _broadcast!(f, C, A, Bs::Vararg{Any,N}) where N
    shape = indices(C)
    @boundscheck check_broadcast_indices(shape, A, Bs...)
    keeps, Idefaults = map_newindexer(shape, A, Bs)
    iter = CartesianRange(shape)
    _broadcast!(f, C, keeps, Idefaults, A, Bs, Val(N), iter)
    return C
end

# broadcast with element type adjusted on-the-fly. This widens the element type of
# B as needed (allocating a new container and copying previously-computed values) to
# accomodate any incompatible new elements.
@generated function _broadcast!(f, B::AbstractArray, keeps::K, Idefaults::ID, As::AT, ::Val{nargs}, iter, st, count) where {K,ID,AT,nargs}
    quote
        $(Expr(:meta, :noinline))
        # destructure the keeps and As tuples
        @nexprs $nargs i->(A_i = As[i])
        @nexprs $nargs i->(keep_i = keeps[i])
        @nexprs $nargs i->(Idefault_i = Idefaults[i])
        while !done(iter, st)
            I, st = next(iter, st)
            # reverse-broadcast the indices
            @nexprs $nargs i->(I_i = newindex(I, keep_i, Idefault_i))
            # extract array values
            @nexprs $nargs i->(@inbounds val_i = _broadcast_getindex(A_i, I_i))
            # call the function
            V = @ncall $nargs f val
            S = typeof(V)
            # store the result
            if S <: eltype(B)
                @inbounds B[I] = V
            else
                # This element type doesn't fit in B. Allocate a new B with wider eltype,
                # copy over old values, and continue
                newB = Base.similar(B, typejoin(eltype(B), S))
                for II in Iterators.take(iter, count)
                    newB[II] = B[II]
                end
                newB[I] = V
                return _broadcast!(f, newB, keeps, Idefaults, As, Val(nargs), iter, st, count+1)
            end
            count += 1
        end
        return B
    end
end

maptoTuple(f) = Tuple{}
maptoTuple(f, a, b...) = Tuple{f(a), maptoTuple(f, b...).types...}

# An element type satisfying for all A:
# broadcast_getindex(
#     resulttype(A),
#     A, indices(A)
# )::_broadcast_getindex_eltype(A)
_broadcast_getindex_eltype(A) = _broadcast_getindex_eltype(resulttype(A), A)
_broadcast_getindex_eltype(::Type{Bottom}, ::Type{T}) where T = Type{T}
_broadcast_getindex_eltype(::Union{Type{Bottom},Type{Any},Type{Nullable}}, A) = typeof(A)
_broadcast_getindex_eltype(::Type, A) = eltype(A)  # Tuple, Array, etc.

# An element type satisfying for all A:
# unsafe_get(A)::unsafe_get_eltype(A)
_unsafe_get_eltype(x::Nullable) = eltype(x)
_unsafe_get_eltype(::Type{T}) where T = Type{T}
_unsafe_get_eltype(x) = typeof(x)

# Inferred eltype of result of broadcast(f, xs...)
_broadcast_eltype(f, A, As...) =
    Base._return_type(f, maptoTuple(_broadcast_getindex_eltype, A, As...))
_nullable_eltype(f, A, As...) =
    Base._return_type(f, maptoTuple(_unsafe_get_eltype, A, As...))

"""
    broadcast(f, As...)

Broadcasts the arrays, tuples, `Ref`s, nullables, and/or scalars `As` to a
container of the appropriate type and dimensions. In this context, anything
that is not a subtype of `AbstractArray`, `Ref` (except for `Ptr`s), `Tuple`,
or `Nullable` is considered a scalar. The resulting container is established by
the following rules:

 - If all the arguments are scalars, it returns a scalar.
 - If the arguments are tuples and zero or more scalars, it returns a tuple.
 - If the arguments contain at least one array or `Ref`, it returns an array
   (expanding singleton dimensions), and treats `Ref`s as 0-dimensional arrays,
   and tuples as 1-dimensional arrays.

The following additional rule applies to `Nullable` arguments: If there is at
least one `Nullable`, and all the arguments are scalars or `Nullable`, it
returns a `Nullable` treating `Nullable`s as "containers".

A special syntax exists for broadcasting: `f.(args...)` is equivalent to
`broadcast(f, args...)`, and nested `f.(g.(args...))` calls are fused into a
single broadcast loop.

# Examples
```jldoctest
julia> A = [1, 2, 3, 4, 5]
5-element Array{Int64,1}:
 1
 2
 3
 4
 5

julia> B = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10

julia> broadcast(+, A, B)
5×2 Array{Int64,2}:
  2   3
  5   6
  8   9
 11  12
 14  15

julia> parse.(Int, ["1", "2"])
2-element Array{Int64,1}:
 1
 2

julia> abs.((1, -2))
(1, 2)

julia> broadcast(+, 1.0, (0, -2.0))
(1.0, -1.0)

julia> broadcast(+, 1.0, (0, -2.0), Ref(1))
2-element Array{Float64,1}:
 2.0
 0.0

julia> (+).([[0,2], [1,3]], Ref{Vector{Int}}([1,-1]))
2-element Array{Array{Int64,1},1}:
 [1, 1]
 [2, 2]

julia> string.(("one","two","three","four"), ": ", 1:4)
4-element Array{String,1}:
 "one: 1"
 "two: 2"
 "three: 3"
 "four: 4"

julia> Nullable("X") .* "Y"
Nullable{String}("XY")

julia> broadcast(/, 1.0, Nullable(2.0))
Nullable{Float64}(0.5)

julia> (1 + im) ./ Nullable{Int}()
Nullable{Complex{Float64}}()
```
"""
broadcast(f, A, Bs...) = broadcast(f,
                                   Result{resulttype(A, Bs...)}(),
                                   A, Bs...)

"""
    broadcast(f, Broadcast.Result{ContainerType}(), As...)

Specify the container-type of the output of a broadcasting operation.
You can specialize such calls as

    function Broadcast.broadcast(f, ::Broadcast.Result{ContainerType,Void,Void}, As...) where ContainerType
        ...
    end
"""
function broadcast(f, ::Result{ContainerType,Void,Void}, A, Bs...) where ContainerType
    ElType = _broadcast_eltype(f, A, Bs...)
    broadcast(f,
              Result{ContainerType,ElType}(combine_indices(A, Bs...)),
              A, Bs...)
end

"""
    broadcast(f, Broadcast.Result{ContainerType,ElType}(indices), As...)

Specify the container-type, element-type, and indices of the output
of a broadcasting operation. You can specialize such calls as

    function Broadcast.broadcast(f, r::Broadcast.Result{ContainerType,ElType,<:Tuple}, As...) where {ContainerType,ElType}
        ...
    end

This variant might be the most convenient specialization for container types
that don't support [`setindex!`](@ref) and therefore can't use [`broadcast!`](@ref).
"""
function broadcast(f, result::Result{ContainerType,ElType,<:Indices}, As...) where {ContainerType,ElType}
    if !Base._isleaftype(ElType)
        return broadcast_nonleaf(f, result, As...)
    end
    dest = similar(f, result, As...)
    broadcast!(f, dest, As...)
end

# default to BitArray for broadcast operations producing Bool, to save 8x space
# in the common case where this is used for logical array indexing; in
# performance-critical cases where Array{Bool} is desired, one can always
# use broadcast! instead.
function broadcast(f, r::Result{BottomArray,Bool}, As...)
    dest = Base.similar(BitArray, indices(r))
    broadcast!(f, dest, As...)
end

# When ElType is not concrete, use narrowing. Use the first element of each input to determine
# the starting output eltype; the _broadcast! method will widen `dest` as needed to
# accomodate later values.
function broadcast_nonleaf(f, r::Result{BottomArray,ElType,<:Indices}, As...) where ElType
    nargs = length(As)
    shape = indices(r)
    iter = CartesianRange(shape)
    if isempty(iter)
        return Base.similar(Array{ElType}, shape)
    end
    keeps, Idefaults = map_newindexer(shape, As)
    st = start(iter)
    I, st = next(iter, st)
    val = f([ _broadcast_getindex(As[i], newindex(I, keeps[i], Idefaults[i])) for i=1:nargs ]...)
    if val isa Bool
        dest = Base.similar(BitArray, shape)
    else
        dest = Base.similar(Array{typeof(val)}, shape)
    end
    dest[I] = val
    return _broadcast!(f, dest, keeps, Idefaults, As, Val(nargs), iter, st, 1)
end

@inline function broadcast(f, r::Result{<:Nullable,Void,Void}, a...)
    nonnull = all(hasvalue, a)
    S = _nullable_eltype(f, a...)
    if Base._isleaftype(S) && null_safe_op(f, maptoTuple(_unsafe_get_eltype,
                                                         a...).types...)
        Nullable{S}(f(map(unsafe_get, a)...), nonnull)
    else
        if nonnull
            Nullable(f(map(unsafe_get, a)...))
        else
            Nullable{nullable_returntype(S)}()
        end
    end
end

broadcast(f, ::Result{Bottom,Void,Void}, a...) = f(a...)

broadcast(f, ::Result{Tuple,Void,Void}, A, Bs...) =
    tuplebroadcast(f, first_tuple(A, Bs...), A, Bs...)
tuplebroadcast(f, ::NTuple{N,Any}, As...) where {N} =
    ntuple(k -> f(tuplebroadcast_getargs(As, k)...), Val(N))
tuplebroadcast(f, ::NTuple{N,Any}, ::Type{T}, As...) where {N,T} =
    ntuple(k -> f(T, tuplebroadcast_getargs(As, k)...), Val(N))
first_tuple(A::Tuple, Bs...) = A
first_tuple(A, Bs...) = first_tuple(Bs...)
tuplebroadcast_getargs(::Tuple{}, k) = ()
tuplebroadcast_getargs(As, k) =
    (_broadcast_getindex(first(As), k), tuplebroadcast_getargs(tail(As), k)...)


"""
    broadcast_getindex(A, inds...)

Equivalent to [`broadcast`](@ref)ing the `inds` arrays to a common size
and returning an array `[A[ks...] for ks in zip(indsb...)]` (where `indsb`
would be the broadcast `inds`). The shape of the output is equal to the shape of each
element of `indsb`.

# Examples
```jldoctest
julia> A = [11 12; 21 22]
2×2 Array{Int64,2}:
 11  12
 21  22

julia> A[1:2, 1:2]
2×2 Array{Int64,2}:
 11  12
 21  22

julia> broadcast_getindex(A, 1:2, 1:2)
2-element Array{Int64,1}:
 11
 22

julia> A[1:2, 2:-1:1]
2×2 Array{Int64,2}:
 12  11
 22  21

julia> broadcast_getindex(A, 1:2, 2:-1:1)
2-element Array{Int64,1}:
 12
 21
 ```
Because the indices are all vectors, these calls are like `[A[i[k], j[k]] for k = 1:2]`
where `i` and `j` are the two index vectors.

```jldoctest
julia> broadcast_getindex(A, 1:2, (1:2)')
2×2 Array{Int64,2}:
 11  12
 21  22

julia> broadcast_getindex(A, (1:2)', 1:2)
2×2 Array{Int64,2}:
 11  21
 12  22

julia> broadcast_getindex(A, [1 2 1; 1 2 2], [1, 2])
2×3 Array{Int64,2}:
 11  21  11
 12  22  22
```
"""
broadcast_getindex(src::AbstractArray, I::AbstractArray...) =
    broadcast_getindex!(Base.similar(Array{eltype(src)}, combine_indices(I...)),
                        src,
                        I...)

@generated function broadcast_getindex!(dest::AbstractArray, src::AbstractArray, I::AbstractArray...)
    N = length(I)
    Isplat = Expr[:(I[$d]) for d = 1:N]
    quote
        @nexprs $N d->(I_d = I[d])
        check_broadcast_indices(indices(dest), $(Isplat...))  # unnecessary if this function is never called directly
        checkbounds(src, $(Isplat...))
        @nexprs $N d->(@nexprs $N k->(Ibcast_d_k = Base.indices(I_k, d) == OneTo(1)))
        @nloops $N i dest d->(@nexprs $N k->(j_d_k = Ibcast_d_k ? 1 : i_d)) begin
            @nexprs $N k->(@inbounds J_k = @nref $N I_k d->j_d_k)
            @inbounds (@nref $N dest i) = (@nref $N src J)
        end
        dest
    end
end

"""
    broadcast_setindex!(A, X, inds...)

Efficient element-by-element setting of the values of `A` in a pattern established by `inds`.
Equivalent to broadcasting the `X` and `inds` arrays to a common size, and then executing

    for (is, js) in zip(zip(indsb), eachindex(Xb))
        A[is...] = Xb[js...]
    end

where `Xb` and `indsb` are the broadcast `X` and `inds`.

See [`broadcast_getindex`](@ref) for examples of the treatment of `inds`.
"""
@generated function broadcast_setindex!(A::AbstractArray, x, I::AbstractArray...)
    N = length(I)
    Isplat = Expr[:(I[$d]) for d = 1:N]
    quote
        @nexprs $N d->(I_d = I[d])
        checkbounds(A, $(Isplat...))
        shape = combine_indices($(Isplat...))
        @nextract $N shape d->(length(shape) < d ? OneTo(1) : shape[d])
        @nexprs $N d->(@nexprs $N k->(Ibcast_d_k = Base.indices(I_k, d) == 1:1))
        if !isa(x, AbstractArray)
            xA = convert(eltype(A), x)
            @nloops $N i d->shape_d d->(@nexprs $N k->(j_d_k = Ibcast_d_k ? 1 : i_d)) begin
                @nexprs $N k->(@inbounds J_k = @nref $N I_k d->j_d_k)
                @inbounds (@nref $N A J) = xA
            end
        else
            X = x
            @nexprs $N d->(shapelen_d = length(shape_d))
            @ncall $N Base.setindex_shape_check X shapelen
            Xstate = start(X)
            @inbounds @nloops $N i d->shape_d d->(@nexprs $N k->(j_d_k = Ibcast_d_k ? 1 : i_d)) begin
                @nexprs $N k->(J_k = @nref $N I_k d->j_d_k)
                x_el, Xstate = next(X, Xstate)
                (@nref $N A J) = x_el
            end
        end
        A
    end
end

############################################################

# x[...] .= f.(y...) ---> broadcast!(f, dotview(x, ...), y...).
# The dotview function defaults to getindex, but we override it in
# a few cases to get the expected in-place behavior without affecting
# explicit calls to view.   (All of this can go away if slices
# are changed to generate views by default.)

Base.@propagate_inbounds dotview(args...) = getindex(args...)
Base.@propagate_inbounds dotview(A::AbstractArray, args...) = view(A, args...)
Base.@propagate_inbounds dotview(A::AbstractArray{<:AbstractArray}, args::Integer...) = getindex(A, args...)


############################################################
# The parser turns @. into a call to the __dot__ macro,
# which converts all function calls and assignments into
# broadcasting "dot" calls/assignments:

dottable(x) = false # avoid dotting spliced objects (e.g. view calls inserted by @view)
dottable(x::Symbol) = !isoperator(x) || first(string(x)) != '.' || x == :.. # don't add dots to dot operators
dottable(x::Expr) = x.head != :$
undot(x) = x
function undot(x::Expr)
    if x.head == :.=
        Expr(:(=), x.args...)
    elseif x.head == :block # occurs in for x=..., y=...
        Expr(:block, map(undot, x.args)...)
    else
        x
    end
end
__dot__(x) = x
function __dot__(x::Expr)
    dotargs = map(__dot__, x.args)
    if x.head == :call && dottable(x.args[1])
        Expr(:., dotargs[1], Expr(:tuple, dotargs[2:end]...))
    elseif x.head == :$
        x.args[1]
    elseif x.head == :let # don't add dots to `let x=...` assignments
        Expr(:let, undot(dotargs[1]), dotargs[2])
    elseif x.head == :for # don't add dots to for x=... assignments
        Expr(:for, undot(dotargs[1]), dotargs[2])
    elseif (x.head == :(=) || x.head == :function || x.head == :macro) &&
           Meta.isexpr(x.args[1], :call) # function or macro definition
        Expr(x.head, x.args[1], dotargs[2])
    else
        head = string(x.head)
        if last(head) == '=' && first(head) != '.'
            Expr(Symbol('.',head), dotargs...)
        else
            Expr(x.head, dotargs...)
        end
    end
end
"""
    @. expr

Convert every function call or operator in `expr` into a "dot call"
(e.g. convert `f(x)` to `f.(x)`), and convert every assignment in `expr`
to a "dot assignment" (e.g. convert `+=` to `.+=`).

If you want to *avoid* adding dots for selected function calls in
`expr`, splice those function calls in with `\$`.  For example,
`@. sqrt(abs(\$sort(x)))` is equivalent to `sqrt.(abs.(sort(x)))`
(no dot for `sort`).

(`@.` is equivalent to a call to `@__dot__`.)

# Examples
```jldoctest
julia> x = 1.0:3.0; y = similar(x);

julia> @. y = x + 3 * sin(x)
3-element Array{Float64,1}:
 3.52441
 4.72789
 3.42336
```
"""
macro __dot__(x)
    esc(__dot__(x))
end

end # module
