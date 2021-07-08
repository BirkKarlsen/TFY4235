using LinearAlgebra, Plots, ProgressMeter, LaTeXStrings
using Distributions, Statistics

# Functions for Problem 2B

# This function computes the transition probabilites for a given set of
# conditions.
function transition_prob(yi, params, Δt)
    Pₛᵢ = 1 - exp(- Δt * params[1] * yi[2] / Int(params[3]))
    Pᵢᵣ = 1 - exp(- Δt / params[2])
    return Pₛᵢ, Pᵢᵣ
end

# This function simply simulates the the Stochastic SIR Model
function solver_SSIR(y0, params, T, Δt)
    Nt = Int(trunc(T/Δt))
    y = Array{Int64, 2}(undef, length(y0), Nt)
    t = LinRange(0, T, Nt)
    Δt = t[2] - t[1]
    y[:,1] = y0

    for i in 2:1:Nt
        Pₛᵢ, Pᵢᵣ = transition_prob(y[:,i-1], params, Δt)
        Δₛᵢ = rand(Binomial(y[1,i-1], Pₛᵢ))
        Δᵢᵣ = rand(Binomial(y[2,i-1], Pᵢᵣ))

        y[1,i] = y[1,i-1] - Δₛᵢ
        y[2,i] = y[2,i-1] + Δₛᵢ - Δᵢᵣ
        y[3,i] = y[3,i-1] + Δᵢᵣ
    end
    return y, t
end

# This functions is for the Stochastic SIR model
function SSIR_model(yi, params, Δt)
    Pₛᵢ, Pᵢᵣ = transition_prob(yi, params, Δt)
    dyi = Array{Int64, 1}(undef, length(yi))

    Δₛᵢ = rand(Binomial(yi[1], Pₛᵢ))
    Δᵢᵣ = rand(Binomial(yi[2], Pᵢᵣ))

    dyi[1] = - Δₛᵢ
    dyi[2] = Δₛᵢ - Δᵢᵣ
    dyi[3] = Δᵢᵣ
    return dyi
end

# This function runs the same simulation several times to get statistics on
# how likely it is that a outbreak will die out by chance.
function compute_probability_for_no_outbreak(Ninf, params, T, Δt, sample_size, ncores)
    Ninfs = [i for i in 1:1:Ninf]
    prob_decay = Array{Float64, 1}(undef, Ninf)
    display("Running...")

    for i in 1:1:Ninf
        Ni = Ninfs[i]
        Ndecays = 0
        y0 = Array{Int64, 1}(undef, 3)
        y0[1] = Int(params[3]) - Ni
        y0[2] = Ni
        y0[3] = 0
        mes = "Computing probability for Ninf = "*string(Ni)*"..."
        @showprogress 1 mes for j in 1:1:sample_size
            a = 0
            y, t = stochastic_solver(y0, params, T, Δt, SSIR_model)
            mean_inf = mean(y[2,:])
            if mean_inf <= Ni
                a += 1
            end
            Ndecays += a
        end
        prob_decay[i] = Ndecays / sample_size
    end
    return Ninfs, prob_decay
end
