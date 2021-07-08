using LinearAlgebra, Plots, ProgressMeter, LaTeXStrings
using Distributions, Statistics
include("stochasticSolver.jl")

# In this problem the params-array is on the form [β τₑ τᵢ rₛ rₐ fₛ fₐ N]
# and the yi array is [S E I Iₐ R]

# This function computes the transition probabilites for the SEIIaR model for
# a given set of conditions.
function transition_prob_SEIIaR(yi, params, Δt)
    Pₛₑ = 1 - exp(-Δt * params[1] * (params[4] * yi[3] + params[5] * yi[4])/params[8])
    Pₑᵢ = params[6] * (1 - exp(- Δt / params[2]))
    Pₑᵢₐ = params[7] * (1 - exp(- Δt / params[2]))
    Pᵢᵣ = 1 - exp(-Δt / params[3])
    Pᵢₐᵣ = 1 - exp(-Δt / params[3])
    return [Pₛₑ, Pₑᵢ, Pₑᵢₐ, Pᵢᵣ, Pᵢₐᵣ]
end

function transition_prob_SEIIaR_ex(yi, params, Δt, Ns)
    Pₛ = Array{Float64, 2}(undef, length(yi[:,1]), length(yi[1,:]))
    Pₛ[1,:] = 1 .- exp.(-Δt .* params[1] .* (params[4] .* transpose(yi[3,:]) .+ params[5] .* transpose(yi[4,:]))./Ns)
    Pₛ[2,:] = params[6] .* (1 - exp(- Δt / params[2])) .* ones(length(yi[1,:]))
    Pₛ[3,:] = params[7] .* (1 - exp(- Δt / params[2])) .* ones(length(yi[1,:]))
    Pₛ[4,:] = (1 - exp(-Δt / params[3])) .* ones(length(yi[1,:]))
    Pₛ[5,:] = (1 - exp(-Δt / params[3])) .* ones(length(yi[1,:]))
    return Pₛ
end

# This function calculates the changes form the probabilites and the
# population given
function SEIIaR_change(yi, Psi)
    dyi = Array{Int64, 1}(undef, length(yi))

    Δₑᵢ, Δₑᵢₐ, Δₑₑ = rand(Multinomial(yi[2], [Psi[2], Psi[3], 1 - Psi[2] - Psi[3]]))
    Δₛₑ = rand(Binomial(yi[1], Psi[1]))
    Δᵢᵣ = rand(Binomial(yi[3], Psi[4]))
    Δᵢₐᵣ = rand(Binomial(yi[4], Psi[5]))

    dyi[1] = - Δₛₑ
    dyi[2] = Δₛₑ - Δₑᵢ - Δₑᵢₐ
    dyi[3] = Δₑᵢ - Δᵢᵣ
    dyi[4] = Δₑᵢₐ - Δᵢₐᵣ
    dyi[5] = Δᵢₐᵣ + Δᵢᵣ
    return dyi
end

function SEIIaR_change_ex(yi, Psi)
    dyi = Array{Int64, 2}(undef, length(yi[:,1,1]), length(yi[1,:,1]))

    Δₑᵢ, Δₑᵢₐ, Δₑₑ = rand.(Multinomial.(yi[2,:], [Psi[2], Psi[3], 1 .- Psi[2] .- Psi[3]]))
    Δₛₑ = rand.(Binomial.(yi[1,:], Psi[1]))
    Δᵢᵣ = rand.(Binomial.(yi[3,:], Psi[4]))
    Δᵢₐᵣ = rand.(Binomial.(yi[4,:], Psi[5]))

    dyi[1,:] = - Δₛₑ
    dyi[2,:] = Δₛₑ .- Δₑᵢ .- Δₑᵢₐ
    dyi[3,:] = Δₑᵢ .- Δᵢᵣ
    dyi[4,:] = Δₑᵢₐ .- Δᵢₐᵣ
    dyi[5,:] = Δᵢₐᵣ .+ Δᵢᵣ
    return dyi
end

# This function performes the SEIIaR model simulation
function SEIIaR_model(yi, params, Δt)
    Pₛₑ, Pₑᵢ, Pₑᵢₐ, Pᵢᵣ, Pᵢₐᵣ = transition_prob_SEIIaR(yi, params, Δt)
    dyi = Array{Int64, 1}(undef, length(yi))

    Δₑᵢ, Δₑᵢₐ, Δₑₑ = rand(Multinomial(yi[2], [Pₑᵢ, Pₑᵢₐ, 1 - Pₑᵢ - Pₑᵢₐ]))
    Δₛₑ = rand(Binomial(yi[1], Pₛₑ))
    Δᵢᵣ = rand(Binomial(yi[3], Pᵢᵣ))
    Δᵢₐᵣ = rand(Binomial(yi[4], Pᵢₐᵣ))

    dyi[1] = - Δₛₑ
    dyi[2] = Δₛₑ - Δₑᵢ - Δₑᵢₐ
    dyi[3] = Δₑᵢ - Δᵢᵣ
    dyi[4] = Δₑᵢₐ - Δᵢₐᵣ
    dyi[5] = Δᵢₐᵣ + Δᵢᵣ
    return dyi
end


# This function optimizes the rₛ such that there is no exponential grow in I.
function bisection_method_rs(target, ra, rb, params, Δt, T, y0, max_it, tol)
    params[4] = ra
    ya, t = stochastic_solver(y0, params, T, Δt, SEIIaR_model)
    Imaxa = maximum(ya[3,:])

    params[4] = rb
    yb, t = stochastic_solver(y0, params, T, Δt, SEIIaR_model)
    Imaxb = maximum(yb[3,:])

    rc = (ra + rb)/2
    params[4] = rc
    yc, t = stochastic_solver(y0, params, T, Δt, SEIIaR_model)
    Imaxc = maximum(yc[3,:])

    err = target - Imaxc

    i = 0
    while abs(err) > tol
        i += 1
        if i > max_it
            display("Failed to converge")
            return rc, Imaxc
        end
        if (target - Imaxc) * (target - Imaxa) < 0
            rb = rc
            Imaxb = Imaxc
        else
            ra = rc
            Imaxa = Imaxc
        end

        rc = (ra + rb)/2
        params[4] = rc
        yc, t = stochastic_solver(y0, params, T, Δt, SEIIaR_model)
        Imaxc = maximum(yc[3,:])
        err = target - Imaxc
        display("iteration "*string(i)*", error is "*string(err))
    end
    return rc, Imaxc
end

# This function runs the same simulation several times to get statistics on
# how likely it is that a outbreak will only be small.
function compute_probability_for_small_outbreak(r_s_res, y0, params, T, Δt, sample_size)
    r_ss = LinRange(0.2, 0.6, r_s_res)
    prob_decay = Array{Float64, 1}(undef, length(r_ss))
    display("Running...")

    for i in 1:1:length(r_ss)
        rs = r_ss[i]
        Ndecays = 0
        mes = "Computing probability for rₛ = "*string(rs)*"..."
        @showprogress 1 mes for j in 1:1:sample_size
            a = 0
            params[4] = r_ss[i]
            y, t = stochastic_solver(y0, params, T, Δt, SEIIaR_model)
            max_inf = maximum(y[3,:])
            if max_inf <= y0[2]
                a += 1
            end
            Ndecays += a
        end
        prob_decay[i] = Ndecays / sample_size
    end
    return r_ss, prob_decay
end
