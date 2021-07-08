using Distributions

# This file contains a general scheme to perform a stochastic evolution of
# certain populations

function stochastic_solver(y0, params, T, Δt, f)
    Nt = Int(trunc(T/Δt))
    y = Array{Int64, 2}(undef, length(y0), Nt)
    t = LinRange(0, T, Nt)
    Δt = t[2] - t[1]
    y[:,1] = y0

    for i in 2:1:Nt
        y[:,i] = y[:,i-1] .+ f(y[:,i-1], params, Δt)
    end
    y, t
end
