# ccall binding for the Qenivo C ABI. Simplex is the native core; other engines
# are the embedded Python bridge. Lubin et al., JuMP 1.0, is the modelling target
# of QenivoMOI.jl, which calls the functions in this file.

module Qenivo

using Libdl

const OPTIMAL = Int32(0)

function library()
    get(ENV, "QENIVO_C_DLL", "libqenivo_c")
end

const _open = Dict{String,Ptr{Cvoid}}()

function symbol(name::Symbol)
    path = library()
    lib = get!(_open, path) do
        Libdl.dlopen(path)
    end
    Libdl.dlsym(lib, name)
end

function check(rc::Int32, handle)
    rc == OPTIMAL && return
    # QENIVO_OK is 0. A nonzero return is an error code, not a solver status.
    msg = unsafe_string(ccall(symbol(:qenivo_last_error_message), Ptr{Cchar}, (Ptr{Cvoid},), handle))
    error("qenivo C ABI failed ($rc): $msg")
end

"""Solve and return (objective, x). `integer` is empty or a 0/1 vector of length n."""
function solve_csr(cost::Vector{Float64}, rowptr::Vector{Int32}, colidx::Vector{Int32}, values::Vector{Float64},
                   row_lower::Vector{Float64}, row_upper::Vector{Float64}, col_lower::Vector{Float64},
                   col_upper::Vector{Float64}; integer::Vector{Int32}=Int32[], engine::String="simplex",
                   name::String="capi")
    rows = Int32(length(rowptr) - 1)
    cols = Int32(length(cost))
    err = Ref{Int32}(0)
    ip = isempty(integer) ? C_NULL : pointer(integer)
    ci = isempty(colidx) ? C_NULL : pointer(colidx)
    ax = isempty(values) ? C_NULL : pointer(values)
    lc = isempty(row_lower) ? C_NULL : pointer(row_lower)
    uc = isempty(row_upper) ? C_NULL : pointer(row_upper)
    handle = ccall(symbol(:qenivo_model_create), Ptr{Cvoid},
        (Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Int32}),
        rows, cols, pointer(rowptr), ci, ax, pointer(cost), lc, uc, pointer(col_lower), pointer(col_upper),
        ip, C_NULL, C_NULL, C_NULL, err)
    handle == C_NULL && error("qenivo_model_create failed ($(err[]))")
    try
        ccall(symbol(:qenivo_set_name), Int32, (Ptr{Cvoid}, Cstring), handle, name)
        rc = ccall(symbol(:qenivo_solve), Int32, (Ptr{Cvoid}, Cstring), handle, engine)
        check(rc, handle)
        obj = Ref{Float64}(0.0)
        ccall(symbol(:qenivo_result_objective), Int32, (Ptr{Cvoid}, Ptr{Float64}), handle, obj) == 0 ||
            error("no objective")
        x = Vector{Float64}(undef, cols)
        ccall(symbol(:qenivo_result_x), Int32, (Ptr{Cvoid}, Ptr{Float64}, Int32), handle, pointer(x), cols)
        return obj[], x
    finally
        ccall(symbol(:qenivo_model_free), Cvoid, (Ptr{Cvoid},), handle)
    end
end

"""min -x s.t. 0 <= x <= 1. The optimum is -1."""
function bounded_lp()
    obj, x = solve_csr([-1.0], Int32[0], Int32[], Float64[], Float64[], Float64[], [0.0], [1.0]; engine="simplex")
    isapprox(x[1], 1.0) || error("expected x = 1, got $x")
    return obj
end

"""min -x-2y s.t. x+y <= 1, x,y binary. The optimum is -2 at (0, 1)."""
function two_binary(; src::String="")
    if !isempty(src)
        ENV["QENIVO_SRC"] = src
    end
    obj, x = solve_csr([-1.0, -2.0], Int32[0, 2], Int32[0, 1], [1.0, 1.0], [-Inf], [1.0], [0.0, 0.0], [1.0, 1.0];
                       integer=Int32[1, 1], engine="milp", name="two_binary")
    (isapprox(x[1], 0.0) && isapprox(x[2], 1.0)) || error("expected (0, 1), got $x")
    return obj
end

end
