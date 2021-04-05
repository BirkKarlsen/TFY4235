using LinearAlgebra, Plots, Distributions
using Distributed, SharedArrays, LaTeXStrings

# Constants
h = 0.01            # timestep
T = 200             # Total time of simulation
t0 = 0              # Initial time
α = 0               # The Gilbert damping constant
γ = 1               # Gyromagnetic ratio
μ = 1               # The magnetic moment
J = 0               # Spin coupling
dz = 0              # Anisotropy constant
B = [0 0 1]         # External magnetic field

params = [α, γ, μ, J, dz]
params = append!(params, B)

# Initial values of the spins
S0 = 1              # Amplitude of the spin
φ = 0               # Azimuthal angle
θ = π / 2           # Polar angle
N = 1               # Number of spins


# Initialzes an Array of Spins in the samme direction.
function initialize_spins_sd(Sl, φ, θ, N)
    y0 = Array{Float64, 2}(undef, 3, N)
    for i in 1:1:N
        y0[:,i] = Sl .* [sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]
    end
    return y0
end

# Initializes an Array of Spins in random directions.
function initialize_spins_rd(Sl, N)
    y0 = Array{Float64, 2}(undef, 3, N)
    for i in 1:1:N
        θi = rand(Uniform(0, π))
        φi = rand(Uniform(0, 2 * π))
        y0[:,i] = Sl .* [sin(θi)*cos(φi), sin(θi)*sin(φi), cos(θi)]
    end
    return y0
end

# Initializes an Array of spins parallell to z-axis with the first spin in some
# general other direction.
function initialize_spins_sdo(Sl, φ, θ, N)
    y0 = Array{Float64, 2}(undef, 3, N)
    y0[:,1] = Sl .* [sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]
    for i in 2:1:N
        y0[:,i] = Sl .* [0,0,1]
    end
    return y0
end


# This is a general Heun algorithm that solves a system of first order
# ordinary differential equations and returns an array with all the
# points for each function in the system and the times.
function Heun_solver_vector(f, h, T, t0, y0)
    N = Int(T/h)
    y = Array{Float64, 2}(undef, N, length(y0))
    t = LinRange(t0, t0 + T, N)
    y[1,:] = y0

    for i in 2:1:N
        yjp = y[i - 1,:] .+ h .* f(t[i-1,:], y[i-1,:])
        y[i,:] = y[i-1,:] .+ (h/2).*(f(t[i-1,:],y[i-1,:]) .+ f(t[i,:], yjp))
    end
    return y, t
end


# We first define the general LLG equation - the equation to be solved in this
# project. Here S, in general, is an array of 3-vectors and each vector
# represents each spin in the spin chain. the varibale dS is its
# time derivative. first index in S is the vector component and second is the
# spinindex.
function LLG_equation(t, S, params)
    α = params[1]
    γ = params[2]
    μ = params[3]
    J = params[4]
    dz = params[5]
    B = (params[6:8])

    C = - γ / (μ * (1 + α^2))

    # Notice that we have assumed only nearest neighbor interactions here.
    dH = -(1/2) .* J .* (circshift(S, (0,1)) .+ circshift(S, (0,-1))) .- μ .* B
    dH[3,:] -= 2 .* dz .* S[3,:]
    dH = -dH

    dS = Array{Float64, 2}(undef, length(S[:,1]), length(S[1,:]))

    # Idea for later: convert all arrays into shared arrays and to parallell
    # computing of this loop.
    for i in 1:1:length(dS[1,:])
        dS[:,i] = C .* (cross(S[:,i], dH[:,i]) .+ α .* cross(S[:,i], cross(S[:,i], dH[:,i])))
    end
    return dS
end


# incremental version of the Heun algorithm.
function Heun_algorithm(f, h, yi, ti, params)
    yjp = Array{Float64, 2}(undef, length(yi[:,1]), length(yi[1,:]))
    yjp = yi .+ h .* f(ti, yi, params)
    return yi .+ (h/2) .* (f(ti + h, yjp, params) .+ f(ti, yi, params))
end


# Incremental verison of the Euler alorithm
function Euler_algorithm(f, h, yi, ti, params)
    return yi .+ h .* f(ti, yi, params)
end


# Solver of the differential equation
function solver(f, alg, params, h, T, t0, y0)
    N = Int(trunc(T/h))
    t = LinRange(t0, t0 + T, N)
    y = Array{Float64, 3}(undef, length(y0[:,1]), length(y0[1,:]), N)
    y[:,:,1] = y0

    for n in 2:1:N
        y[:,:,n] = alg(f, h, y[:,:,n-1], t[n-1], params)
    end
    return y, t
end


# This function makes an animation of the spins.
function make_animation_one_spin(y)
    n = length(y[1,1,:])
    gr()
    anim = @animate for i ∈ 1:5:n
        quiver([0], [0], [0], quiver=([y[1,1,i]], [y[2,1,i]], [y[3,1,i]]), projection="3d",
        ylims=(-2,2), xlims=(-2,2), zlims=(-2,2))
    end

    gif(anim, "SingleSpin.gif", fps=30)
end


# This function makes an animation of the spins.
function make_animation_mult_spins(y, nl)
    n = length(y[1,1,:])
    pos = zeros(Float64, 3, length(y[1,:,1]))
    pos[1,:] = [i for i in 0:1:(length(y[1,:,1])-1)]

    gr()
    anim = @animate for i ∈ 1:nl:n
        quiver(pos[1,:], pos[2,:], pos[3,:], quiver=(y[1,:,i], y[2,:,i], y[3,:,i]),
        projection="3d", xlims=(-2, length(y[1,:,1]) + 2), ylims=(-2,2), zlims=(-2,2),
        arrow=:filled)
    end
    gif(anim, "TenSpins.gif", fps=30)
end


# Analytic solution of spin in magnetic field
function Analytic_spin_precession(t, S0, θ, γ, B)
    St = Array{Float64, 2}(undef, 3, length(t))
    St[1,:] = S0 * sin(θ) .* cos.(γ .* B .* t)
    St[2,:] = S0 * sin(θ) .* sin.(γ .* B .* t)
    St[3,:] = S0 * cos(θ) .* ones(length(t))
    return St
end


# Finding the error for between the analytic solution of the spin precession
# and the simulated one.
function estimate_error_with_analytic(yS, yA, T, tend)
    nend = Int(trunc(length(yA[1,:]) * tend / T))
    return norm(yA[:,nend] .- yS[:,nend])
end



# These are the main functions for the project


# This function simulates a single spin precession in a magnetic field
# and makes an animation.
function animate_spin_precession(S0, φ, θ, params, T)
    y0 = initialize_spins_sd(S0, φ, θ, 1)
    y, t = solver(LLG_equation, Heun_algorithm, params, 0.01, T, 0, y0)

    make_animation_one_spin(y)
end


# This function simulates a single spin precession in a magnetic field
# and plots the x and y component.
function plot_spin_precession(S0, φ, θ, params, T)
    y0 = initialize_spins_sd(S0, φ, θ, 1)
    y, t = solver(LLG_equation, Heun_algorithm, params, 0.01, T, 0, y0)
    tt = [t t]
    yy = [y[1,1,:] y[2,1,:]]
    plot(tt, yy, label=[L"S_x (t)" L"S_y (t)"], title=L"\textrm{Single spin precession}",
    ylims=(-1,1))
    xlabel!(L"\gamma B_0  t \; [-]")
    ylabel!(L"S_0 \frac{2}{\hbar} \; [-]")
    savefig("SingleSpinPrecession.pdf")
end

function plot_spin_with_analytic_solution(S0, φ, θ, params, T)
    y0 = initialize_spins_sd(S0, φ, θ, 1)
    y, t = solver(LLG_equation, Heun_algorithm, params, 0.01, T, 0, y0)
    yA = Analytic_spin_precession(t, S0, θ, params[2], params[8])
    yy1 = [y[1,1,:] y[2,1,:]]
    yy2 = [yA[1,:] yA[2,:]]
    tt = [t t]

    p1 = plot(tt, yy1, title=L"\textrm{Numerical}")
    p2 = plot(tt, yy2, title=L"\textrm{Analytic}")
    p3 = plot(p1, p2, layout = (2,1), ylims=(-1,1), label = [L"S_x(t)" L"S_y(t)"])
    xlabel!(L"\gamma B_0  t \; [-]")
    ylabel!(L"S_0 \frac{2}{\hbar} \; [-]")
    savefig(p3, "SpinWithAnalyticSolution.pdf")
end

function plot_spin_damped_precession(S0, φ, θ, params, T)
    y0 = initialize_spins_sd(S0, φ, θ, 1)
    y, t = solver(LLG_equation, Heun_algorithm, params, 0.01, T, 0, y0)
    decay = S0 * sin(θ) .* exp.(-t .* abs(params[1] * params[8] * params[2]))
    tt = [t t t]
    yy = [y[1,1,:] y[2,1,:] decay]
    plot(tt, yy, label = [L"$S_x (t)$" L"$S_y (t)$" L"\textrm{Theoretical decay}"],
    title=L"\textrm{Single spin damped precession}")
    xlabel!(L"\gamma B_0  t \; [-]")
    ylabel!(L"S_0 \frac{2}{\hbar} \; [-]")
    savefig("SingleSpinDampedPrecession.pdf")
end


# This function computes and calculates the error in the simulation
function error_analysis_single_spin(S0, φ, θ, params, T, tend)
    y0 = initialize_spins_sd(S0, φ, θ, 1)
    hs = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    errors = Array{Float64, 2}(undef, 5, 2)

    for i in 1:1:5
        ySH, t = solver(LLG_equation, Heun_algorithm, params, hs[i], T, 0, y0)
        ySE, t = solver(LLG_equation, Euler_algorithm, params, hs[i], T, 0, y0)
        yA = Analytic_spin_precession(t, S0, θ, γ, params[8])
        errors[i, 1] = estimate_error_with_analytic(ySH[:,1,:], yA, T, tend)
        errors[i, 2] = estimate_error_with_analytic(ySE[:,1,:], yA, T, tend)
    end
    hss = [hs hs]

    scatter(hs, errors, shape=:x, xaxis=:log, yaxis=:log, label=[L"\textrm{Heun}" L"\textrm{Euler}"],
    title=L"\textrm{Error as a function of timestep}", legend=:topleft)
    xlabel!(L"h \; [\textrm{s}]")
    ylabel!(L"\Delta S \; [\textrm{N} \textrm{m}^2 \textrm{/ s}]")
    savefig("ErrorComparison.pdf")
end


# This function looks at the time evolution of a ten-spin array with random
# initialized spins.
function plot_random_spins(S0, params, T, h)
    y0 = initialize_spins_rd(S0, 10)
    y, t = solver(LLG_equation, Heun_algorithm, params, h, T, 0, y0)

    tt = [t t t t t t t t t t]
    yy = [y[3,1,:] y[3,2,:] y[3,3,:] y[3,4,:] y[3,5,:] y[3,6,:] y[3,7,:] y[3,8,:] y[3,9,:] y[3,10,:]]
    plot(tt, yy, title=L"\textrm{Ten spins in random directions,} \; J < 0",
    linecolor=[1 2 1 2 1 2 1 2 1 2], label = [L"S_{z,1} (t)" L"S_{z,2} (t)" nothing nothing nothing nothing nothing nothing nothing nothing])
    xlabel!(L"\gamma B_0  t \; [-]")
    ylabel!(L"S_0 \frac{2}{\hbar} \; [-]")
    savefig("RandomSpinsGSN.pdf")
end

function animate_random_spins(S0, params, T, h, nl, N)
    y0 = initialize_spins_rd(S0, N)
    y, t = solver(LLG_equation, Heun_algorithm, params, h, T, 0, y0)
    display("Plotting...")
    make_animation_mult_spins(y, nl)
end


# This function looks at the time evolution of a ten-spin array with all spins
# initialized in same directions
function plot_spins_sd(S0, φ, θ, params, T, h)
    y0 = initialize_spins_sd(S0, φ, θ, 10)
    y, t = solver(LLG_equation, Heun_algorithm, params, h, T, 0, y0)

    tt = [t t t t t t t t t t]
    yy = [y[1,1,:] y[1,2,:] y[1,3,:] y[1,4,:] y[1,5,:] y[1,6,:] y[1,7,:] y[1,8,:] y[1,9,:] y[1,10,:]]
    plot(tt, yy, title=L"\textrm{Ten tilted spins with} \; J = 0", ylims=(-1,1),
    label = [L"S_{x,1} (t)" L"S_{x,2} (t)" L"S_{x,3} (t)" L"S_{x,4} (t)" L"S_{x,5} (t)" L"S_{x,6} (t)" L"S_{x,7} (t)" L"S_{x,8} (t)" L"S_{x,9} (t)" L"S_{x,10} (t)"],
    legend=:bottomright)
    xlabel!(L"\gamma B_0  t \; [-]")
    ylabel!(L"S_0 \frac{2}{\hbar} \; [-]")
    savefig("TenTiltedSpinsNC.pdf")
end

function animate_spins_sd(S0, φ, θ, params, T, h, nl, N)
    y0 = initialize_spins_sd(S0, φ, θ, N)
    y, t = solver(LLG_equation, Heun_algorithm, params, h, T, 0, y0)
    display("Plotting...")
    make_animation_mult_spins(y, nl)
end

# This function looks at the time evolution of a ten-spin array with all spins
# parallell to the z-axis, except for the first, which is in some general
# direction.
function plot_spins_sdo(S0, φ, θ, params, T, h)
    y0 = initialize_spins_sdo(S0, φ, θ, 10)
    y, t = solver(LLG_equation, Heun_algorithm, params, h, T, 0, y0)

    tt = [t t t t t t t t t t]
    yy = [y[3,1,:] y[3,2,:] y[3,3,:] y[3,4,:] y[3,5,:] y[3,6,:] y[3,7,:] y[3,8,:] y[3,9,:] y[3,10,:]]
    plot(tt, yy, title=L"\textrm{Ten spins with only one tilted,} \; J > 0 \textrm{ and } \alpha > 0",
    legend = :outertopright, ylims=(0,1),
    label = [L"S_{z,1} (t)" L"S_{z,2} (t)" L"S_{z,3} (t)" L"S_{z,4} (t)" L"S_{z,5} (t)" L"S_{z,6} (t)" L"S_{z,7} (t)" L"S_{z,8} (t)" L"S_{z,9} (t)" L"S_{z,10} (t)"],
    linestyle=[:solid :solid :solid :solid :solid :dot :dot :dot :dot :dot],
    linealpha=[0.4 0.4 0.4 0.4 0.4 1 1 1 1 1])
    xlabel!(L"\gamma B_0  t \; [-]")
    ylabel!(L"S_0 \frac{2}{\hbar} \; [-]")
    savefig("TenSpinsOneTiltedCWithDampingZ.pdf")
end

function animate_spins_sdo(S0, φ, θ, params, T, h, nl, N)
    y0 = initialize_spins_sdo(S0, φ, θ, N)
    y, t = solver(LLG_equation, Heun_algorithm, params, h, T, 0, y0)
    display("Plotting...")
    make_animation_mult_spins(y, nl)
end


# This function simulates the spin precession with the damping term
#plot_spin_precession(S0, φ, θ, params, T)
#plot_spin_damped_precession(S0, φ, θ, params, T)

# Error calculations
#plot_spin_with_analytic_solution(S0, φ, θ, params, T)
#error_analysis_single_spin(S0, φ, θ, params, T, 100)

# This function simulates and plots the z-component of 10 coupled spins
# the structure of the params array is [α, γ, μ, J, dz]
params = [0.05, γ, μ, -1, 0.1]
params = append!(params, [0 0 0])

#animate_random_spins(S0, params, T, h, 20, 100)
#plot_random_spins(S0, params, T, h)

# This function simulates and plots the x-component of 10 spins precessing
params = [0.05, γ, μ, 0.1, 0.1]
params = append!(params, [0 0 0])

#animate_spins_sdo(S0, φ, θ, params, T, h, 40, 100)
#plot_spins_sdo(S0, φ, θ, params, T, h)
