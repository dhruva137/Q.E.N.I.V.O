# MathOptInterface wrapper. JuMP talks to this optimizer; optimize! builds CSR and
# calls the C ABI in Qenivo.jl. Legat, Dowson, Garcia, Lubin, MathOptInterface.

module QenivoMOI

using MathOptInterface
const MOI = MathOptInterface

include(joinpath(@__DIR__, "Qenivo.jl"))

mutable struct Optimizer <: MOI.AbstractOptimizer
    sense::MOI.OptimizationSense
    lower::Vector{Float64}
    upper::Vector{Float64}
    cost::Vector{Float64}
    integer::Vector{Int32}
    rows::Vector{Vector{Tuple{Int,Float64}}}
    row_lower::Vector{Float64}
    row_upper::Vector{Float64}
    x::Vector{Float64}
    offset::Float64
    objective::Float64
    termination::MOI.TerminationStatusCode
end

function Optimizer()
    Optimizer(MOI.FEASIBILITY_SENSE, Float64[], Float64[], Float64[], Int32[],
              Vector{Tuple{Int,Float64}}[], Float64[], Float64[], Float64[], 0.0, NaN,
              MOI.OPTIMIZE_NOT_CALLED)
end

function MOI.empty!(opt::Optimizer)
    empty!(opt.lower); empty!(opt.upper); empty!(opt.cost); empty!(opt.integer)
    empty!(opt.rows); empty!(opt.row_lower); empty!(opt.row_upper); empty!(opt.x)
    opt.sense = MOI.FEASIBILITY_SENSE
    opt.offset = 0.0
    opt.objective = NaN
    opt.termination = MOI.OPTIMIZE_NOT_CALLED
end

MOI.is_empty(opt::Optimizer) = isempty(opt.lower) && isempty(opt.rows)

function MOI.add_variable(opt::Optimizer)
    # MOI variables are free until a bound constraint is added.
    push!(opt.lower, -Inf)
    push!(opt.upper, Inf)
    push!(opt.cost, 0.0)
    push!(opt.integer, Int32(0))
    return MOI.VariableIndex(length(opt.lower))
end

function _set_bound(opt, vi::MOI.VariableIndex, lo, hi)
    i = vi.value
    if lo !== nothing
        opt.lower[i] = lo
    end
    if hi !== nothing
        opt.upper[i] = hi
    end
    return MOI.ConstraintIndex{MOI.VariableIndex,Any}(i)
end

function MOI.add_constraint(opt::Optimizer, f::MOI.VariableIndex, s::MOI.GreaterThan{Float64})
    _set_bound(opt, f, s.lower, nothing)
end
function MOI.add_constraint(opt::Optimizer, f::MOI.VariableIndex, s::MOI.LessThan{Float64})
    _set_bound(opt, f, nothing, s.upper)
end
function MOI.add_constraint(opt::Optimizer, f::MOI.VariableIndex, s::MOI.EqualTo{Float64})
    _set_bound(opt, f, s.value, s.value)
end
function MOI.add_constraint(opt::Optimizer, f::MOI.VariableIndex, ::MOI.ZeroOne)
    opt.integer[f.value] = Int32(1)
    _set_bound(opt, f, 0.0, 1.0)
end
function MOI.add_constraint(opt::Optimizer, f::MOI.VariableIndex, ::MOI.Integer)
    opt.integer[f.value] = Int32(1)
    return MOI.ConstraintIndex{MOI.VariableIndex,MOI.Integer}(f.value)
end

function _row(opt, f::MOI.ScalarAffineFunction{Float64}, lo, hi)
    terms = Tuple{Int,Float64}[(t.variable.value, t.coefficient) for t in f.terms]
    # Move the function constant onto the bounds: constant + a'x in [lo, hi].
    shift = f.constant
    push!(opt.rows, terms)
    push!(opt.row_lower, lo === nothing ? -Inf : lo - shift)
    push!(opt.row_upper, hi === nothing ? Inf : hi - shift)
    return MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},Any}(length(opt.rows))
end

MOI.add_constraint(opt::Optimizer, f::MOI.ScalarAffineFunction{Float64}, s::MOI.LessThan{Float64}) =
    _row(opt, f, nothing, s.upper)
MOI.add_constraint(opt::Optimizer, f::MOI.ScalarAffineFunction{Float64}, s::MOI.GreaterThan{Float64}) =
    _row(opt, f, s.lower, nothing)
MOI.add_constraint(opt::Optimizer, f::MOI.ScalarAffineFunction{Float64}, s::MOI.EqualTo{Float64}) =
    _row(opt, f, s.value, s.value)

function MOI.set(opt::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    opt.sense = sense
end

function MOI.set(opt::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
                 f::MOI.ScalarAffineFunction{Float64})
    fill!(opt.cost, 0.0)
    for t in f.terms
        opt.cost[t.variable.value] += t.coefficient
    end
    opt.offset = f.constant
end

const _AFFINE_SETS = Union{MOI.LessThan{Float64}, MOI.GreaterThan{Float64}, MOI.EqualTo{Float64}}
const _VAR_SETS = Union{MOI.LessThan{Float64}, MOI.GreaterThan{Float64}, MOI.EqualTo{Float64}, MOI.ZeroOne, MOI.Integer}

MOI.supports_constraint(::Optimizer, ::Type{MOI.VariableIndex}, ::Type{<:_VAR_SETS}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{<:_AFFINE_SETS}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

function MOI.optimize!(opt::Optimizer)
    n = length(opt.cost)
    m = length(opt.rows)
    rowptr = Int32[0]
    colidx = Int32[]
    values = Float64[]
    for terms in opt.rows
        for (j, a) in terms
            push!(colidx, Int32(j - 1))
            push!(values, a)
        end
        push!(rowptr, Int32(length(colidx)))
    end
    if m == 0
        rowptr = Int32[0]
    end
    cost = opt.sense == MOI.MAX_SENSE ? -opt.cost : copy(opt.cost)
    engine = any(!iszero, opt.integer) ? "milp" : "simplex"
    obj, x = Qenivo.solve_csr(cost, rowptr, colidx, values, collect(opt.row_lower), collect(opt.row_upper),
                              collect(opt.lower), collect(opt.upper); integer=opt.integer, engine=engine)
    opt.x = x
    sign = opt.sense == MOI.MAX_SENSE ? -1.0 : 1.0
    opt.objective = sign * obj + opt.offset
    opt.termination = MOI.OPTIMAL
end

MOI.get(opt::Optimizer, ::MOI.TerminationStatus) = opt.termination
MOI.get(opt::Optimizer, ::MOI.PrimalStatus) = isempty(opt.x) ? MOI.NO_SOLUTION : MOI.FEASIBLE_POINT
MOI.get(opt::Optimizer, ::MOI.ObjectiveValue) = opt.objective
MOI.get(opt::Optimizer, ::MOI.VariablePrimal, vi::MOI.VariableIndex) = opt.x[vi.value]

end
