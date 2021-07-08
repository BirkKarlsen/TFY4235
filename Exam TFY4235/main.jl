using LinearAlgebra, Plots, ProgressMeter, LaTeXStrings
using Distributions, DelimitedFiles

# Include the other files with functions for each subproblem
include("problem2A.jl")
include("problem2B.jl")
include("problem2C.jl")
include("problem2D.jl")
include("stochasticSolver.jl")


# Problem 2A - The deterministic SIR model

I_N = 1e-4

# Initializes the parameters for the SIR model
params = Array{Float64, 1}(undef, 3)
params[1] = 0.25        # β [days⁻¹]
params[2] = 10          # τ [days]
params[3] = 1           # N [%]

ℛ₀ = params[1] * params[2]

# Initial condition
y0 = Array{Float64, 1}(undef, 3)
y0[1] = params[3] - I_N * params[3]
y0[2] = I_N * params[3]
y0[3] = 0

# Parameters for the simulation
t0 = 0                  # [days]
T = 180                 # [days]
h = 0.01                # [days]

# Task a)
prob2A_taska = false
if prob2A_taska
    y, t = solver(deterministic_SIR, RK4_algorithm, params, h, T, t0, y0)
    S_inf, R_inf = asymptotic_SR(ℛ₀, params[3], 0.5, 0.5, 1000)
    S_inf = S_inf .* ones(length(t))
    R_inf = R_inf .* ones(length(t))

    plot(t, y[1,:], label=L"S_D(t)", legend=:right,
    title=L"\textrm{The Deterministic SIR Model}")
    plot!(t, y[2,:], label=L"I_D(t)")
    plot!(t, y[3,:], label=L"R_D(t)")
    plot!(t, S_inf, label=L"S(\infty)", linestyle=[:dot], linecolor=1)
    plot!(t, R_inf, label=L"R(\infty)", linestyle=[:dot], linecolor=3)
    xlabel!(L"t \; [\textrm{days}]")
    ylabel!(L"\textrm{Fraction of population }[-]")
    savefig("problem2Aa.pdf")
end

# Task b)
prob2A_taskb = false
if prob2A_taskb
    y, t = solver(deterministic_SIR, RK4_algorithm, params, h, T, t0, y0)
    I_an = analytic_I(y0, t)

    plot(t, y[2,:], label=L"I_D(t)", yaxis=:log, legend=:right,
    title=L"\textrm{Infecion from DSIR model with approximate solution}")
    plot!(t, I_an, label=L"I_A(t)")
    xlabel!(L"t \; [\textrm{days}]")
    ylabel!(L"\textrm{Fraction of population}[-]")
    savefig("problem2Ab.pdf")
end

# Task c)
prob2A_taskc = false
if prob2A_taskc
    βs, I_max = optimize_β(params, h, T, y0, 200)
    plot(βs, I_max, label=L"I_{\textrm{max}}(\beta)", legend=:right,
    title=L"\textrm{Maximum infected as a function of }\beta")
    tol = 1e-6
    βc, Imaxc = bisection_method_β(0.2-tol, 0.1, 0.5, deterministic_SIR, RK4_algorithm, params, h, T, t0, y0, 100, tol)
    display(βc)
    display(Imaxc)
    scatter!([βc], [Imaxc], shape=:x,
    label=L"I_{\textrm{max}} ( \beta_{\textrm{crit}} )")
    ylabel!(L"I_{\textrm{max}} \; [-]")
    xlabel!(L"\beta \; [1/\textrm{days}]")
    savefig("problem2Ac.pdf")
end

# Task d)
prob2A_taskd = false
if prob2A_taskd
    vaccs, I_max = optimize_vacc(params, h, T, y0, 200)
    plot(vaccs, I_max, label=L"I_{\textrm{max}}(R(0))", legend=:right,
    title=L"\textrm{Maximum infected as a function of }R(0)")
    y0 = Array{Float64, 1}(undef, 3)
    y0[1] = params[3] - I_N * params[3]
    y0[2] = I_N * params[3]
    y0[3] = 0
    tol = 1e-16

    vacc_max, I_maxc = bisection_method_vacc(y0[2] - tol, 0.1, 0.9, deterministic_SIR, RK4_algorithm, params, h, T, t0, y0, 100, tol)
    display(vacc_max)
    display(I_maxc)
    scatter!([vacc_max], [I_maxc], shape=:x,
    label=L"I_{\textrm{max}} ( R_{\textrm{crit}} )")
    ylabel!(L"I_{\textrm{max}} \; [-]")
    xlabel!(L"R(0) \; [-]")
    savefig("problem2Ad.pdf")
end



# Problem 2B - The Stochastic SIR Model

# Constants

β = 0.25            # [days⁻¹]
τ = 10              # [days]
N = 100000          # population size [-]
T = 180             # total time [days]
Δt = 0.01           # time step [days]

# Task a)
prob2B_taska = false
if prob2B_taska
    Ninf = 10
    params = [β τ N]

    y0 = Array{Int64, 1}(undef, 3)
    y0[1] = Int(params[3]) - Ninf
    y0[2] = Ninf
    y0[3] = 0

    run_and_plot = false
    if run_and_plot
        y, t = solver(deterministic_SIR, RK4_algorithm, params, h, T, t0, y0)
        plot([t t t], transpose(y), label=[L"S_D(t)" L"I_D(t)" L"R_D(t)"],
        legend=:right, title=L"\textrm{SSIR and DSIR models}")

        for i in 1:1:10
            y, t = stochastic_solver(y0, params, T, Δt, SSIR_model)
            if i == 1
                plot!([t t t], transpose(y), linestyle=:dot, linecolor=[6 4 5],
                label=[L"S_S(t)" L"I_S(t)" L"R_S(t)"], linealpha=[0.6 0.6 0.6])
            else
                plot!([t t t], transpose(y), linestyle=:dot, linecolor=[6 4 5],
                label=[nothing nothing nothing], linealpha=[0.6 0.6 0.6])
            end
        end
        xlabel!(L"t \; [\textrm{days}]")
        ylabel!(L"\textrm{Amount of people }[-]")
        savefig("problem2Ba.pdf")
    end

    error_an = false
    if error_an
        y, t = solver(deterministic_SIR, RK4_algorithm, params, h, T, t0, y0)
        y_D_max = maximum(y[2,:])

        Δts = [1, 0.1, 0.01, 0.001, 0.0001]
        N_trials = 50
        plot(xaxis = :log, title=L"\textrm{Difference in }I_{\textrm{max}}\textrm{ between DSIR and SSIR}")
        @showprogress 1 "Computing..." for i ∈ 1:1:length(Δts)
            I_maxes_diff = zeros(N_trials)
            for j ∈ 1:1:N_trials
                y, t = stochastic_solver(y0, params, T, Δts[i], SSIR_model)
                I_maxes_diff[j] = y_D_max - maximum(y[2,:])
            end
            σ = [(std(I_maxes_diff), std(I_maxes_diff))]
            if i == 1
                scatter!([Δts[i]], [mean(I_maxes_diff)], yerror=σ,
                label=L"\textrm{Mean with standard deviation}", color=2)
            else
                scatter!([Δts[i]], [mean(I_maxes_diff)], yerror=σ,
                label=nothing, color=2)
            end
        end
        ylabel!(L"I_{D}^{\textrm{max}} - I_{S}^{\textrm{max}} \; [-]")
        xlabel!(L"\Delta t \; [\textrm{days}]")
        savefig("problem2Ba2.pdf")
    end
end

# Task b)
prob2B_taskb = false
if prob2B_taskb
    Ninf = 10
    params = [β τ N]

    y0 = Array{Int64, 1}(undef, 3)
    y0[1] = Int(params[3]) - Ninf
    y0[2] = Ninf
    y0[3] = 0
    t = LinRange(0, T, 1000)
    y = analytic_I(y0, t)
    plot(t, y, yaxis=:log, legend=:right, label=L"I_A(t)",
    title=L"\textrm{Infection SSIR model with approximate solution}")

    for i in 1:1:10
        y, t = stochastic_solver(y0, params, T, Δt, SSIR_model)
        if i == 1
            plot!(t, y[2,:], linestyle=:dot, linecolor=6,
            label=L"I_S(t)", linealpha=0.6)
        else
            plot!(t, y[2,:], linestyle=:dot, linecolor=6,
            label=nothing, linealpha=0.6)
        end
    end
    xlabel!(L"t \; [\textrm{days}]")
    ylabel!(L"\textrm{Amount of people }[-]")
    savefig("problem2Bb.pdf")
end

# Task c)
prob2B_taskc = false
if prob2B_taskc
    Ninf = 10
    params = [β τ N]

    y0 = Array{Int64, 1}(undef, 3)
    y0[1] = Int(params[3]) - Ninf
    y0[2] = Ninf
    y0[3] = 0

    run = false
    if run
        Ninfs_matrix = zeros(10, 10)
        prob_decay_matrix = zeros(10, 10)
        for i ∈ 1:1:10
            Ninfs, prob_decay = compute_probability_for_no_outbreak(Ninf, params, T, Δt, 10000, 8)
            display(Ninfs)
            display(prob_decay)
            Ninfs_matrix[:,i] = Ninfs
            prob_decay_matrix[:,i] = prob_decay
        end
        writedlm("problem2Bc_data_Ninfs_matrix.csv", Ninfs_matrix, ',')
        writedlm("problem2Bc_data_prob_decay_matrix.csv", prob_decay_matrix, ',')
    end

    plot_ = false
    if plot_
        Ninfs_matrix = readdlm("problem2Bc_data_Ninfs_matrix.csv", ',')
        prob_decay_matrix = readdlm("problem2Bc_data_prob_decay_matrix.csv", ',')
        means_prob = mean(prob_decay_matrix, dims=2)
        std_prob = std(prob_decay_matrix, dims=2)

        #scatter(Ninfs[:,1], means_prob, yerror=σ)
        p1 = plot(Ninfs_matrix[:,1], means_prob, label=nothing,
        title=L"\textrm{Probability that the epidemic will die out by chance}")
        ylabel!(L"\textrm{Mean probability }[-]")
        xlabel!(L"I_S(0) \; [-]")
        p2 = plot(Ninfs_matrix[:,1], std_prob, label=nothing)
        ylabel!(L"\textrm{Standard derivation }[-]")
        xlabel!(L"I_S(0) \; [-]")
        plot(p1, p2, layout=(2,1))
        savefig("problem2Bc.pdf")
    end
end



# Problem 2C - The Stochastic SEIIaR Model

# In this problem the params-array is on the form [β τₑ τᵢ rₛ rₐ fₛ fₐ N]
# and the yi array is [S E I Iₐ R]

# Constants for the simulation

β = 0.55            # [days⁻¹]
rₛ = 1.0            # [-]
rₐ = 0.1            # [-]
fₛ = 0.6            # [-]
fₐ = 0.4            # [-]
τₑ = 3              # [days]
τᵢ = 7              # [days]

T = 180             # [days]
Δt = 0.01           # [days]
N = 100000          # population size [-]

# Task a)
prob2C_taska = false
if prob2C_taska
    params = [β τₑ τᵢ rₛ rₐ fₛ fₐ N]
    Ne = 25

    y0 = Array{Int64, 1}(undef, 5)
    y0[1] = N - Ne
    y0[2] = Ne
    y0[3:end] .= 0

    params1 = [0.25 10 100000]
    y01 = Array{Float64, 1}(undef, 3)
    y01[1] = params1[3] - I_N * params1[3]
    y01[2] = I_N * params1[3]
    y01[3] = 0
    y, t = solver(deterministic_SIR, RK4_algorithm, params1, h, T, t0, y01)
    plot([t t t], transpose(y), label=[L"S_D(t)" L"I_D(t)" L"R_D(t)"], legend=:right,
    title=L"\textrm{The SEIIaR and DSIR models}")

    for i in 1:1:10
        y, t = stochastic_solver(y0, params, T, Δt, SEIIaR_model)
        Y = y[1:end-1,:]
        Y[3,:] = y[3,:] .+ y[4,:]
        Y[4,:] = y[5,:]
        if i == 1
            plot!([t t t t t], transpose(y), linestyle=:dot, linecolor=[4 1 5 6 7],
            label=[L"S_S(t)" L"E_S(t)" L"I_S(t)" L"I_{aS}(t)" L"R_S(t)"],
            linealpha=[0.6 0.6 0.6 0.6 0.6])
        else
            plot!([t t t t t], transpose(y), linestyle=:dot, linecolor=[4 1 5 6 7],
            label=[nothing nothing nothing nothing nothing],
            linealpha=[0.6 0.6 0.6 0.6 0.6])
        end
    end
    xlabel!(L"t \; [\textrm{days}]")
    ylabel!(L"\textrm{Amount of people }[-]")
    savefig("problem2Ca.pdf")
end

# Task b)
prob2C_taskb = false
if prob2C_taskb
    params = [β τₑ τᵢ rₛ rₐ fₛ fₐ N]
    Ne = 25
    params[4] = 0.4

    y0 = Array{Int64, 1}(undef, 5)
    y0[1] = N - Ne
    y0[2] = Ne
    y0[3:end] .= 0

    r_s_res = 20
    experiments = 10

    run_code = false
    if run_code
        prob_matrix = Array{Float64, 2}(undef, r_s_res, experiments)
        r_ss_matrix = Array{Float64, 2}(undef, r_s_res, experiments)
        for i ∈ 1:1:experiments
            r_ss, prob_decay = compute_probability_for_small_outbreak(r_s_res, y0, params, T, Δt, 1000)
            prob_matrix[:,i] = prob_decay
            r_ss_matrix[:,i] = r_ss
        end
        writedlm("problem2Cb_data_r_ss_matrix.csv", r_ss_matrix, ',')
        writedlm("problem2Cb_data_prob_decay_matrix.csv", prob_matrix, ',')
    end

    plot_code = false
    if plot_code
        prob_matrix = readdlm("problem2Cb_data_prob_decay_matrix.csv", ',')
        r_ss_matrix = readdlm("problem2Cb_data_r_ss_matrix.csv", ',')
        mean_prob = mean(prob_matrix, dims=2)
        std_prob = std(prob_matrix, dims=2)

        plot(r_ss_matrix[:,1], mean_prob, ribbon=std_prob,
        title=L"\textrm{Probability for only a small outbreak}",
        label=L"P\;(I_{\textrm{max}} < E(0))")
        xlabel!(L"r_s \; [-]")
        ylabel!(L"\textrm{Probability } [-]")
        savefig("problem2Cb.pdf")
    end

    #y, t = stochastic_solver(y0, params, T, Δt, SEIIaR_model)
    #plot([t t t t t], transpose(y))
    #tol = 1e-6

    #rs, Imax = bisection_method_rs(25, 0.01, 0.9, params, Δt, T, y0, 100, tol)
    #display(rs)
    #display(Imax)
end



# Problem 2D - The Stochastic SEIIaR commuter model

# Constants for the simulation

β = 0.55            # [days⁻¹]
rₛ = 1.0            # [-]
rₐ = 0.1            # [-]
fₛ = 0.6            # [-]
fₐ = 0.4            # [-]
τₑ = 3              # [days]
τᵢ = 7              # [days]

T = 180             # [days]
Δt = 0.01           # [days]

# The population for Town 1 and Town 2:
Mₚ = [9000 1000; 200 99800]


# Task a)
prob2D_taska = false
if prob2D_taska
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

    y, t = commuter_simulation(y0, params, T , Δt)

    y_each_town = sum(y, dims=3)[:,:,1,:]
    display(sum(y_each_town[:,2,1]))
    p1 = plot([t t t t t], transpose(y_each_town[:,1,:]),
    legend=:outerright, title=L"\textrm{The SEIIaR model for Town 1}",
    label=[L"S(t)" L"E(t)" L"I(t)" L"I_a(t)" L"R(t)"])
    xlabel!(L"t \; [\textrm{days}]")
    ylabel!(L"\textrm{Amount of people }[-]")


    p2 = plot([t t t t t], transpose(y_each_town[:,2,:]),
    legend=:outerright, title=L"\textrm{The SEIIaR model for Town 2}",
    label=[L"S(t)" L"E(t)" L"I(t)" L"I_a(t)" L"R(t)"])
    xlabel!(L"t \; [\textrm{days}]")
    ylabel!(L"\textrm{Amount of people }[-]")


    plot(p1, p2, layout=(2,1))

    savefig("problem2Da.pdf")
end

# Task b)

# This is an explanation of the code



# Problem 2E - Larger stochastic SEIIaR commuter simulations

# Constants

β = 0.55            # [days⁻¹]
rₛ = 1.0            # [-]
rₐ = 0.1            # [-]
fₛ = 0.6            # [-]
fₐ = 0.4            # [-]
τₑ = 3              # [days]
τᵢ = 7              # [days]

T = 180             # [days]
Δt = 0.01           # [days]


# Task a)
prob2E_taska = true
if prob2E_taska
    params = [β τₑ τᵢ rₛ rₐ fₛ fₐ]
    Ne = 25

    # Initializing the population matrix
    M = zeros(Int64, 10, 10)
    M[1,1] = 198600
    M[1,2:5] .= 100
    M[2:5,1] .= 500
    M[1,6], M[6,1] = 1000, 1000
    M[2,2], M[3,3], M[4,4], M[5,5] = 9500, 9500, 9500, 9500
    M[6,6] = 498200
    M[6,7:end] .= 200
    M[7:end,6] .= 1000
    M[7,7], M[8,8], M[9,9], M[10, 10] =  19000,  19000,  19000,  19000

    display(M)
    display(reduce_travel(M, 0.9))

    # Initializing the initial condition
    y0 = Array{Int64, 3}(undef, 5, length(M[:,1]), length(M[1,:]))
    y0 .= 0
    y0[1,:,:] = M
    y0[1,2,2] -= Ne
    y0[2,2,2] += Ne


    y, t = commuter_simulation(y0, params, T, Δt)
    display(params)
    y_each_town_matrix = Array{Int64, 4}(undef, 5, length(M[:,1]), length(t), 10)

    y_each_town = sum(y, dims=3)[:,:,1,:]
    plots_array = Array{Any, 1}(undef, length(y_each_town[1,:,1]))
    for i ∈ 1:1:10
        y, t = commuter_simulation(y0, params, T, Δt)
        y_each_town = sum(y, dims=3)[:,:,1,:]
        y_each_town_matrix[:,:,:,i] = y_each_town
    end

    for i ∈ 1:1:length(y_each_town[1,:,1])
        pli = plot()
        for j ∈ 1:1:10
            if j == 1
                plot!([t t t t t], transpose(y_each_town_matrix[:,i,:,j]),
                label = [L"S(t)" L"E(t)" L"I(t)" L"I_a(t)" L"R(t)"],
                linestyle=:dot, linecolor=[4 1 5 6 7])
            else
                plot!([t t t t t], transpose(y_each_town_matrix[:,i,:,j]),
                label = [nothing nothing nothing nothing nothing],
                linestyle=:dot, linecolor=[4 1 5 6 7])
            end
        end
        plots_array[i] = pli
    end

    plot(plots_array[1], plots_array[2], plots_array[3],
    plots_array[4], plots_array[5], plots_array[6], plots_array[7],
    plots_array[8], plots_array[9], plots_array[10], layout=(5,2),
    size=(1000, 1000), legend=:outerright,
    title=[L"\textrm{Town 1}" L"\textrm{Town 2}" L"\textrm{Town 3}" L"\textrm{Town 4}" L"\textrm{Town 5}" L"\textrm{Town 6}" L"\textrm{Town 7}" L"\textrm{Town 8}" L"\textrm{Town 9}" L"\textrm{Town 10}"])


    savefig("problem2Ea.pdf")
end

# Task b)
prob2E_taskb = false
if prob2E_taskb
    params = [β τₑ τᵢ rₛ rₐ fₛ fₐ]
    Ne = 50

    M = readdlm("population_structure.csv", ',', Int64)

    # Initializing the initial condition
    y0 = Array{Int64, 3}(undef, 5, length(M[:,1]), length(M[1,:]))
    y0 .= 0
    y0[1,:,:] = M
    y0[1,1,1] -= Ne
    y0[2,1,1] += Ne
    println(Threads.nthreads())
    display(M)

    run_code = false
    if run_code
        inf_matrix = Array{Int64, 2}(undef, Int(trunc(T/Δt)), 10)
        for i ∈ 1:1:10
            y, t = commuter_simulation(y0, params, T, Δt)

            y_each_town = sum(y, dims=3)[:,:,1,:]
            N_infected = count_infected(y_each_town, 10)
            inf_matrix[:,i] = N_infected
        end
        writedlm("infected_matrix_1.csv", inf_matrix, ',')
    end

    read_code = false
    if read_code
        inf_matrix = readdlm("infected_matrix.csv", ',', Int64)
        t = LinRange(0, T, length(inf_matrix[:,1]))
        plot(t, inf_matrix[:,1], linestyle=:dot, label=nothing,
        title=L"\textrm{The municipalities with } I + I_a > 10")
        for i ∈ 2:1:length(inf_matrix[1,:])
            plot!(t, inf_matrix[:,i], linestyle=:dot, label=nothing)
        end
        ylabel!(L"\textrm{Amount of municipalities with } I + I_a > 10 \; [-]")
        xlabel!(L"t \; [\textrm{days}]")
        savefig("problem2Eb.pdf")
    end
end

# Task c)
prob2E_taskc = false
if prob2E_taskc
    params = [β τₑ τᵢ rₛ rₐ fₛ fₐ]
    Ne = 50

    M = readdlm("population_structure.csv", ',', Int64)
    display(M)
    Mr = reduce_travel(M, 0.9)
    display(Mr)
    display(sum(M))
    display(sum(Mr))

    # Initializing the initial condition
    y0 = Array{Int64, 3}(undef, 5, length(Mr[:,1]), length(Mr[1,:]))
    y0 .= 0
    y0[1,:,:] = Mr
    y0[1,1,1] -= Ne
    y0[2,1,1] += Ne
    println(Threads.nthreads())

    run_code = false
    if run_code
        inf_matrix = Array{Int64, 2}(undef, Int(trunc(T/Δt)), 10)
        for i ∈ 1:1:10
            y, t = commuter_simulation(y0, params, T, Δt)

            y_each_town = sum(y, dims=3)[:,:,1,:]
            N_infected = count_infected(y_each_town, 10)
            inf_matrix[:,i] = N_infected
        end
        writedlm("infected_matrix_reduced_1.csv", inf_matrix, ',')
    end

    read_code = false
    if read_code
        inf_matrix = readdlm("infected_matrix_reduced.csv", ',', Int64)
        t = LinRange(0, T, length(inf_matrix[:,1]))
        plot(t, inf_matrix[:,1], linestyle=:dot, label=nothing,
        title=L"\textrm{The municipalities with } I + I_a > 10 \textrm{ and } r_t = 0.9")
        for i ∈ 2:1:length(inf_matrix[1,:])
            plot!(t, inf_matrix[:,i], linestyle=:dot, label=nothing)
        end
        ylabel!(L"\textrm{Amount of municipalities with } I + I_a > 10 \; [-]")
        xlabel!(L"t \; [\textrm{days}]")
        savefig("problem2Ec.pdf")
    end
end
