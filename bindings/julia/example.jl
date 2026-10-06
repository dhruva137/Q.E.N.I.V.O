# Run from this directory: julia example.jl
# QENIVO_C_DLL must point at the C ABI shared library.
# QENIVO_SRC must point at qenivo/src so the embedded interpreter can import the MILP engine.

include(joinpath(@__DIR__, "Qenivo.jl"))

obj = Qenivo.bounded_lp()
println("optimum ", obj)
@assert isapprox(obj, -1.0)

milp = Qenivo.two_binary(src=get(ENV, "QENIVO_SRC", ""))
println("milp ", milp)
@assert isapprox(milp, -2.0)
