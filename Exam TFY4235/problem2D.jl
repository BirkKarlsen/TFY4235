using LinearAlgebra, Plots, ProgressMeter, LaTeXStrings
using Distributions, Statistics, Distributed, FLoops
include("problem2C.jl")
include("stochasticSolver.jl")

# the Stochastic SEIIaR commuter model

# The population for Town 1 and Town 2:
Mₚ = [9000 1000; 200 99800]

# This function performes each time step of the simulation.
function stochastic_timestep(yis, params, Δt, day)
    dyi = Array{Int64, 3}(undef, length(yis[:,1,1]), length(yis[1,:,1]), length(yis[1,1,:]))

    if day % 2 == 0     # night
        pops = sum(yis, dims=3)[:,:,1]
    else                # day
        pops = sum(yis, dims=2)[:,1,:]
    end

    Pₛ = Array{Float64, 2}(undef, length(pops[:,1]), length(pops[1,:]))
    Ns = sum(pops, dims=1)
    Pₛ = transition_prob_SEIIaR_ex(pops, params, Δt, Ns)

    if day % 2 == 0     # night
        for i in 1:1:length(yis[1,:,1])
            for j in 1:1:length(yis[1,1,:])
                dyi[:,i,j] = SEIIaR_change(yis[:,i,j], Pₛ[:,i])
            end
        end
    else                # day
        for i in 1:1:length(yis[1,:,1])
            for j in 1:1:length(yis[1,1,:])
                dyi[:,i,j] = SEIIaR_change(yis[:,i,j], Pₛ[:,j])
            end
        end
    end
    return yis .+ dyi
end

function stochastic_timestep_ex(yis, params, Δt, day)
    dyi = Array{Int64, 3}(undef, length(yis[:,1,1]), length(yis[1,:,1]), length(yis[1,1,:]))

    if day % 2 == 0     # night
        pops = sum(yis, dims=3)[:,:,1]
    else                # day
        pops = sum(yis, dims=2)[:,1,:]
    end

    Ns = sum(pops, dims=1)
    Pₛ = transition_prob_SEIIaR_ex(pops, params, Δt, Ns)

    if day % 2 == 0     # night
        for i ∈ 1:1:length(yis[1,:,1])
            dyi[:,i,:] = SEIIaR_change_ex(yis[:,i,:], Pₛ[:,i])
        end
    else                # day
        for i ∈ 1:1:length(yis[1,1,:])
            #Pₛₘ = permutedims(Pₛₘ, [1, 3, 2])
            dyi[:,:,i] = SEIIaR_change_ex(yis[:,:,i], Pₛ[:,i])
        end
    end
    return yis .+ dyi
end

function stochastic_timestep_p(yis, params, Δt, day)
    dyi = Array{Int64, 3}(undef, length(yis[:,1,1]), length(yis[1,:,1]), length(yis[1,1,:]))

    if day % 2 == 0     # night
        pops = sum(yis, dims=3)[:,:,1]
    else                # day
        pops = sum(yis, dims=2)[:,1,:]
    end

    Pₛ = Array{Float64, 2}(undef, length(pops[:,1]), length(pops[1,:]))
    Ns = sum(pops, dims=1)
    Pₛ = transition_prob_SEIIaR_ex(pops, params, Δt, Ns)

    if day % 2 == 0     # night
        Threads.@threads for i in 1:1:length(yis[1,:,1])
            for j in 1:1:length(yis[1,1,:])
                dyi[:,i,j] = SEIIaR_change(yis[:,i,j], Pₛ[:,i])
            end
        end
    else                # day
        Threads.@threads for i in 1:1:length(yis[1,:,1])
            for j in 1:1:length(yis[1,1,:])
                dyi[:,i,j] = SEIIaR_change(yis[:,i,j], Pₛ[:,j])
            end
        end
    end
    return yis .+ dyi
end

# This function simulates the SEIIaR commuter model
function commuter_simulation(y0, params, T, Δt)
    day = 0
    hc = 0

    Nt = Int(trunc(T/Δt))
    t = LinRange(0, T, Nt)
    Δt = t[2] - t[1]

    y = Array{Int64, 4}(undef, length(y0[:,1,1]), length(y0[1,:,1]), length(y0[1,1,:]), Nt)
    y[:,:,:,1] = y0
    display("Running...")

    @showprogress 1 "Simulating the commuter model..." for i in 2:1:Nt
        y[:,:,:,i] = stochastic_timestep(y[:,:,:,i-1], params, Δt, day)

        hc += Δt
        if hc > 0.5
            day += 1
            hc = 0
        end
    end
    return y, t
end


function commuter_simulation_ex(y0, params, T, Δt)
    day = 0
    hc = 0

    Nt = Int(trunc(T/Δt))
    t = LinRange(0, T, Nt)
    Δt = t[2] - t[1]

    y = Array{Int64, 4}(undef, length(y0[:,1,1]), length(y0[1,:,1]), length(y0[1,1,:]), Nt)
    y[:,:,:,1] = y0
    display("Running...")

    @showprogress 1 "Simulating the commuter model..." for i in 2:1:Nt
        y[:,:,:,i] = stochastic_timestep_ex(y[:,:,:,i-1], params, Δt, day)

        hc += Δt
        if hc > 0.5
            day += 1
            hc = 0
        end
    end
    return y, t
end

function commuter_simulation_p(y0, params, T, Δt)
    day = 0
    hc = 0

    Nt = Int(trunc(T/Δt))
    t = LinRange(0, T, Nt)
    Δt = t[2] - t[1]

    y = Array{Int64, 4}(undef, length(y0[:,1,1]), length(y0[1,:,1]), length(y0[1,1,:]), Nt)
    y[:,:,:,1] = y0
    display("Running...")

    @showprogress 1 "Simulating the commuter model..." for i in 2:1:Nt
        y[:,:,:,i] = stochastic_timestep_p(y[:,:,:,i-1], params, Δt, day)

        hc += Δt
        if hc > 0.5
            day += 1
            hc = 0
        end
    end
    return y, t
end

# This function reduces travel in the commuter matrix
function  reduce_travel(M, re)
    for i ∈ 1:1:length(M[:,1])
        row1 = M[i,1:i-1] .-  Int.(round.((1 - re) .* M[i,1:i-1], digits=0))
        row2 = M[i,i+1:end] .-  Int.(round.((1 - re) .* M[i,i+1:end], digits=0))

        M[i,1:i-1] = Int.(round.((1 - re) .* M[i,1:i-1], digits=0))
        M[i,i+1:end] = Int.(round.((1 - re) .* M[i,i+1:end], digits=0))
        M[i,i] += sum(row1) + sum(row2)
    end
    return M
end

function count_infected(y_towns, thres)
    N_infected = zeros(Int64, length(y_towns[1,1,:]))

    for i ∈ 1:1:length(y_towns[1,1,:])
        I_towns_ti = y_towns[3:4,:,i]
        I_towns_ti = sum(I_towns_ti, dims=1)[1,:]
        N_infected[i] = length(I_towns_ti[I_towns_ti .> thres])
    end
    return N_infected
end

# Testing:

β = 0.55            # [days⁻¹]
rₛ = 1.0            # [-]
rₐ = 0.1            # [-]
fₛ = 0.6            # [-]
fₐ = 0.4            # [-]
τₑ = 3              # [days]
τᵢ = 7              # [days]

T = 180             # [days]
Δt = 0.01           # [days]


params = [β τₑ τᵢ rₛ rₐ fₛ fₐ]

Ne = 25
y0 = Array{Int64, 3}(undef, 5, length(Mₚ[:,1]), length(Mₚ[1,:]))
y0[1,:,:] = Mₚ
y0[2:end,:,:] .= 0


Ns = sum(y0[1,:,:], dims=2)
ps = y0[1,1,:] ./ Ns[1]
Δ = rand(Multinomial(Ne, ps))
y0[1,1,:] = y0[1,1,:] .- Δ
y0[2,1,:] = Δ

Ne = 25
y0 = Array{Int64, 3}(undef, 5, length(Mₚ[:,1]), length(Mₚ[1,:]))
y0[1,:,:] = Mₚ
y0[2:end,:,:] .= 0

y0[1,1,1] -= Ne
y0[2,1,1] += Ne

#y,  t = commuter_simulation(y0, params, T, Δt)
#y,  t = commuter_simulation_p(y0, params, T, Δt)

#y_each_town = sum(y, dims=3)[:,:,1,:]
#display(sum(y_each_town[:,2,1]))
#p1 = plot([t t t t t], transpose(y_each_town[:,1,:]))
#p2 = plot([t t t t t], transpose(y_each_town[:,2,:]))
#plot(p1, p2, layout=(2,1))
