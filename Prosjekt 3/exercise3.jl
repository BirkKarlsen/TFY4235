using Plots, LinearAlgebra, LaTeXStrings, SparseArrays, ProgressMeter
using DelimitedFiles, Statistics

# Physical Constants

H = 5060            # Concentration proportionality constant [mol m^-3 atm^-1]
a = 6.97e-7         # Mass transfer windspeed coefficient [s/m]
pCO2 = 415e-6       # Partial pressure of CO2 [atm]
u = 10              # Constant average windspeed [m/s]
l = 100             # Depth of the ocean [m]

# Numerical constants

Δt = 0.05            # Time step [s]
Δz = 0.0005         # Spatial step [m]
τ = 1000             # Total time of the simulation [s]

# Relations

Ceq = H * pCO2
kw = a * u^2

l = 4000
τ = 60 * 60 * 24 * 365 * 10
Δt = 60 * 60 * 12
Δz = 0.4

# Functions

# Initializing the matrices in the simulation
function initialize_L(α, Γ, K, Kᴬ)
    du = (-α / 4) .* Kᴬ[1: end-1] .- α .* K[1: end-1]
    dl = (α / 4) .* Kᴬ[2 : end] .- α .* K[2 : end]
    d = 1 .+ (2 * α) .* K
    du[1] = -2 * α * K[1]
    dl[end] = -2 * α * K[end]
    d[1] += Γ

    return Tridiagonal(dl, d, du)
end

function initialize_R(α, Γ, K, Kᴬ)
    du = (α / 4) .* Kᴬ[1 : end-1] .+ α .* K[1 : end-1]
    dl = (-α / 4) .* Kᴬ[2 : end] .+ α .* K[2 : end]
    d = 1 .- 2 * α .* K
    du[1] = 2 * α * K[1]
    dl[end] = 2 * α * K[end]
    d[1] -= Γ

    return Tridiagonal(dl, d, du)
end

function initialize_LR(α, Γ, K)
    N = length(K)
    duL = Array{Float64, 1}(undef, N - 1)
    dL = Array{Float64, 1}(undef, N)
    dlL = Array{Float64, 1}(undef, N - 1)

    duR = Array{Float64, 1}(undef, N - 1)
    dR = Array{Float64, 1}(undef, N)
    dlR = Array{Float64, 1}(undef, N - 1)

    for i in 1:1:N
        dL[i] = 1 + 2 * α * K[i]
        dR[i] = 1 - 2 * α * K[i]

        if i < N && i > 1
            duL[i] = - α * K[i] - (α / 4) * (K[i + 1] - K[i - 1])
            duR[i] = α * K[i] + (α / 4) * (K[i + 1] - K[i - 1])
        end
        if i < N - 1
            dlL[i] = - α * K[i + 1] + (α / 4) * (K[i + 2] - K[i])
            dlR[i] = + α * K[i + 1] - (α / 4) * (K[i + 2] - K[i])
        end
    end
    dL[1] += Γ
    dR[1] -= Γ

    duL[1] = - 2 * α * K[1]
    duR[1] = + 2 * α * K[1]

    dlL[end] = - 2 * α * K[end]
    dlR[end] = + 2 * α * K[end]

    return Tridiagonal(dlL, dL, duL), Tridiagonal(dlR, dR, duR)
end

# Implementation of the TDMA algorithm
function TDMA(L, d)
    N = size(L)[1]
    a = diag(L, -1)
    b = diag(L, 0)
    c = diag(L, 1)
    cₘ = Array{Float64, 1}(undef, N - 1)
    dₘ = Array{Float64, 1}(undef, N)
    x = Array{Float64, 1}(undef, N)

    cₘ[1] = c[1]/b[1]
    dₘ[1] = d[1]/b[1]
    for i in 2:1:N
        if i != N
            cₘ[i] = c[i] / (b[i] - a[i - 1]*cₘ[i - 1])
        end

        dₘ[i] = (d[i] - a[i - 1] * dₘ[i - 1])/(b[i] - a[i - 1] * cₘ[i - 1])
    end
    x[end] = dₘ[end]
    for j in 1:1:N-1
        i = N - j
        x[i] = dₘ[i] - cₘ[i] * x[i + 1]
    end
    return x
end

# This function iterates forward in time using the Crank-Nicolson Method.
function time_iteration(F, R, Cⁱ, S, i)
    V = R * Cⁱ .+ (1/2) .* (S[:,i] .+ S[:,i + 1])
    return F \ V
end

# This function solves the problem.
function solver(K_func, kw, C₀, St, Δt, Δz, l, τ)
    Nt = Int(trunc(τ / Δt)) + 1
    Nz = Int(trunc(l / Δz)) + 1
    z = LinRange(0,l,Nz)
    t = LinRange(0,τ,Nt)
    Δt = t[2] - t[1]
    Δz = z[2] - z[1]

    K = K_func(z)
    C = Array{Float64,2}(undef, Nz, Nt)
    C[:,1] = C₀(z)
    S = spzeros(Nz,Nt)
    S[1,:] = St

    α = Δt / (2 * Δz^2)
    Γ = 2 * α * kw * Δz * (1 - (-(3/2)*K[1] + 2 * K[2] - (1/2) * K[3])/(2 * K[1]))
    Kᴬ = -(circshift(K, 1) .- circshift(K, -1))

    L = initialize_L(α, Γ, K, Kᴬ)
    R = initialize_R(α, Γ, K, Kᴬ)
    S = 2 .* Γ .* S
    F = lu(L)

    @showprogress 1 "Computing..." for i in 1:1:(Nt-1)
        C[:,i + 1] = time_iteration(F, R, C[:,i], S, i)
    end
    return C, t, z, K
end

function solver_TDMA(K_func, kw, C₀, St, Δt, Δz, l, τ)
    Nt = Int(trunc(τ / Δt)) + 1
    Nz = Int(trunc(l / Δz)) + 1
    z = LinRange(0,l,Nz)
    t = LinRange(0,τ,Nt)
    Δt = t[2] - t[1]
    Δz = z[2] - z[1]

    K = K_func(z)
    C = Array{Float64,2}(undef, Nz, Nt)
    C[:,1] = C₀(z)
    S = spzeros(Nz,Nt)
    S[1,:] = St

    α = Δt / (2 * Δz^2)
    Γ = 2 * α * kw * Δz * (1 - (-(3/2)*K[1] + 2 * K[2] - (1/2) * K[3])/(2 * K[1]))

    L, R = initialize_LR(α, Γ, K)
    Kᴬ = -(circshift(K, 1) .- circshift(K, -1))

    #L = initialize_L(α, Γ, K, Kᴬ)
    #R = initialize_R(α, Γ, K, Kᴬ)
    S = 2 .* Γ .* S

    @showprogress 1 "Computing..." for i in 1:1:(Nt-1)
        Vi = R * C[:,i] .+ (1/2) .* (S[:,i] .+ S[:,i + 1])
        C[:,i + 1] = TDMA(L, Vi)
    end
    return C, t, z, K
end

# Numerical integration:
function simps(f, x)
    Δx = x[2] - x[1]
    s = f[1,:] .+ f[end,:]
    s = s .+ 4 .* transpose(sum(f[2:2:end-1,:], dims=1))
    s = s .+ 2 .* transpose(sum(f[3:2:end-1,:], dims=1))
    return (Δx/3) .* s[:,1]
end

function compute_μ(C, z)
    Cz = Array{Float64, 2}(undef, length(C[:,1]), length(C[1,:]))
    @showprogress 1 "Computing μ..." for i in 1:1:length(C[1,:])
        Cz[:,i] = C[:,i] .* z
    end
    return simps(Cz, z) ./ simps(C, z)
end


# Plotting functions
function plot_mass(C, z, t)
    Δz = z[2] - z[1]
    M = simps(C, z)

    plot(t, (100 .* (M[1] .- M)./M[1]), label=L"M_{rel}(t)",
    title=L"\textrm{Total mass as a function of time}")
    xlabel!(L"t \; [\textrm{s}]")
    ylabel!(L"M_{rel} \; [\textrm{%}]")
    savefig("MassConservation.pdf")
end

function animate_massdistribution(C, z, t, nl)
    gr()
    anim = @animate for i in 1:nl:length(t)
        plot(C[:,i], z, label=L"C(z,t)", xlims=(0,3),
        title=L"\textrm{Mass Distribution}")
        yaxis!(yflip=true)
        xlabel!(L"C(z,t)")
        ylabel!(L"z \; [\textrm{m}]")
    end
    gif(anim, "DistributionAnimation2.gif", fps=30)
end

function plot_error_for_Δt(Δts_t, μt_error, RMSt_error, fn, a1, a2)
    p1 = scatter(Δts_t, μt_error, shape=:x, xaxis=:log, yaxis=:log,
    legend = :topleft, label=nothing)
    plot!(Δts_t, a1 .* Δts_t.^2, linestyle=:dot, label=L"\sim \Delta t^2")
    xlabel!(L"\Delta t \; [\textrm{s}]")
    ylabel!(L"\textrm{Error 1st moment}")

    p2 = scatter(Δts_t, RMSt_error, shape=:x, xaxis=:log, yaxis=:log,
    legend = :topleft, label=nothing)
    plot!(Δts_t, a2 .* Δts_t.^2, linestyle=:dot, label=L"\sim \Delta t^2")
    xlabel!(L"\Delta t \; [\textrm{s}]")
    ylabel!(L"\textrm{Error RMS moment}")

    plot(p1, p2, layout=2, size=(800,400))
    savefig(fn)
end

function plot_error_for_Δz(Δzs_z, μz_error, RMSz_error, fn, a1, a2)
    p1 = scatter(Δzs_z, μz_error, shape=:x, xaxis=:log, yaxis=:log,
    legend = :topleft, label=nothing)
    plot!(Δzs_z, a1 .* Δzs_z.^2, linestyle=:dot, label=L"\sim \Delta z^2")
    xlabel!(L"\Delta z \; [\textrm{s}]")
    ylabel!(L"\textrm{Error 1st moment}")

    p2 = scatter(Δzs_z, RMSz_error, shape=:x, xaxis=:log, yaxis=:log,
    legend = :topleft, label=nothing)
    plot!(Δzs_z, a2 .* Δzs_z.^2, linestyle=:dot, label=L"\sim \Delta z^2")
    xlabel!(L"\Delta z \; [\textrm{s}]")
    ylabel!(L"\textrm{Error RMS moment}")

    plot(p1, p2, layout=2, size=(800,400))
    savefig(fn)
end

# Functions for error analysis
function estimate_error_time_sw(kw, Ceq, timesteps, Δz, l, τ)
    timesteps = τ .* timesteps
    N = length(timesteps)
    μ_error = Array{Float64, 1}(undef, N - 1)
    RMS_error = Array{Float64, 1}(undef, N - 1)

    St = Ceq .* ones(Int(trunc(τ / timesteps[end])) + 1)
    C_ref, t_ref, z, K = solver_TDMA(K₃, kw, C03, St, timesteps[end], Δz, l, τ)
    μ_ref = compute_μ(C_ref[:,end:end], z)[end]

    for i in 1:1:(N - 1)
        St = Ceq .* ones(Int(trunc(τ / timesteps[i])) + 1)
        C, t, z, K = solver_TDMA(K₃, kw, C03, St, timesteps[i], Δz, l, τ)
        μ_error[i] = abs(compute_μ(C[:,end:end], z)[end] - μ_ref)
        RMS_error[i] = sqrt(mean((C[:,end] .- C_ref[:,end]).^2))
    end

    return timesteps[1:end-1], μ_error, RMS_error
end

function estimate_error_space_sw(kw, Ceq, spacesteps, Δt, l, τ)
    spacesteps = l .* spacesteps
    N = length(spacesteps)
    μ_error = Array{Float64, 1}(undef, N - 1)
    RMS_error = Array{Float64, 1}(undef, N - 1)

    St = Ceq .* ones(Int(trunc(τ / Δt)) + 1)
    C_ref, t_ref, z, K = solver_TDMA(K₃, kw, C03, St, Δt, spacesteps[end], l, τ)
    μ_ref = compute_μ(C_ref[:,end:end], z)[end]
    Nx_ref = length(C_ref[:,1])

    for i in 1:1:(N - 1)
        C, t, z, K = solver_TDMA(K₃, kw, C03, St, Δt, spacesteps[i], l, τ)
        μ_error[i] = abs(compute_μ(C[:,end:end], z)[end] - μ_ref)
        Nx = length(C[:,1])
        Nxskip = Int(trunc((Nx_ref - 1) / (Nx - 1)))
        RMS_error[i] = sqrt(mean((C[:,end] .- C_ref[1:Nxskip:end,end]).^2))
    end
    return spacesteps[1:end-1], μ_error, RMS_error
end

function estimate_error_time_dw(kw, Ceq, ΔCeq, timesteps, Δz, l, τ)
    timesteps = τ .* timesteps
    N = length(timesteps)
    μ_error = Array{Float64, 1}(undef, N - 1)
    RMS_error = Array{Float64, 1}(undef, N - 1)

    Nt = Int(trunc(τ / timesteps[end])) + 1
    St = Ceq .* ones(Nt) .+ ΔCeq .* LinRange(0, τ, Nt)
    C_ref, t_ref, z, K = solver_TDMA(K₄, kw, C03, St, timesteps[end], Δz, l, τ)
    μ_ref = compute_μ(C_ref[:,end:end], z)[end]

    for i in 1:1:(N - 1)
        Nt = Int(trunc(τ / timesteps[i])) + 1
        St = Ceq .* ones(Nt) .+ ΔCeq .* LinRange(0, τ, Nt)
        C, t, z, K = solver_TDMA(K₄, kw, C03, St, timesteps[i], Δz, l, τ)
        μ_error[i] = abs(compute_μ(C[:,end:end], z)[end] - μ_ref)
        RMS_error[i] = sqrt(mean((C[:,end] .- C_ref[:,end]).^2))
    end

    return timesteps[1:end-1], μ_error, RMS_error
end

function estimate_error_space_dw(kw, Ceq, ΔCeq, spacesteps, Δt, l, τ)
    spacesteps = l .* spacesteps
    N = length(spacesteps)
    μ_error = Array{Float64, 1}(undef, N - 1)
    RMS_error = Array{Float64, 1}(undef, N - 1)

    Nt = Int(trunc(τ / Δt)) + 1
    St = Ceq .* ones(Nt) .+ ΔCeq .* LinRange(0, τ, Nt)
    C_ref, t_ref, z_ref, K = solver_TDMA(K₄, kw, C03, St, Δt, spacesteps[end], l, τ)
    μ_ref = compute_μ(C_ref[:,end:end], z_ref)[end]
    Nx_ref = length(C_ref[:,1])

    for i in 1:1:(N - 1)
        C, t, z, K = solver_TDMA(K₄, kw, C03, St, Δt, spacesteps[i], l, τ)
        μ_error[i] = abs(compute_μ(C[:,end:end], z)[end] - μ_ref)
        Nx = length(C[:,1])
        Nxskip = Int(trunc((Nx_ref - 1) / (Nx - 1)))
        RMS_error[i] = sqrt(mean((C[:,end] .- C_ref[1:Nxskip:end,end]).^2))
    end
    return spacesteps[1:end-1], μ_error, RMS_error
end

# Functions for K(z)

function K₁(z)
    K₀ = 1e-3
    return K₀ .* exp.(-z / l)
end

function K₂(z)
    K₀ = 1e-3
    return K₀ .* ones(length(z))
end

function K₃(z)
    K₀ = 1e-3           # [m^2/s]
    Kₐ = 2e-2           # [m^2/s]
    zₐ = 7              # [m]
    Kᵦ = 5e-2           # [m^2/s]
    zᵦ = 10             # [m]
    l = 100             # [m]
    return K₀ .+ Kₐ .* (z ./ zₐ) .* exp.(.- z ./ zₐ) .+ Kᵦ .* ((l .- z)./zᵦ) .* exp.(.-(l .- z)./zᵦ)
end

function K₄(z)
    K₀ = 1e-4           # [m¨2/s]
    K₁ = 1e-2           # [m^2/s]
    a = 0.5             # [m^-1]
    z₀ = 100            # [m]
    return K₁ .+ (K₀ - K₁) ./ (1 .+ exp.(.- a .* (z .- z₀)))
end

# Some Functions for the initial mass distributions.
function C01(z)
    l = 100         # [m]
    return exp.(.-z./l)
end

function C02(z)
    l = 100         # [m]
    return exp.(.-(z .- l/2).^2 ./(2 * 1^2))
end

function C03(z)
    C₀ = Ceq
    #C₀ = 0
    return C₀ .* ones(length(z))
end


# Running tht code

# The problems:

# Problem 1

# Test case 1 - The well-mixed condition
test1 = false
if test1
    # Δz = 0.001, Δt = 100, τ = 60 * 60 * 24 * 7, l = 100
    St = zeros(Int(trunc(τ / Δt)) + 1)
    C, t, z, K = solver_TDMA(K₃, 0, C03, St, Δt, Δz, l, τ)
    p1 = plot(C[:,end], z, xlims=(0,2), label=L"C(z)",
    title=L"\textrm{Final concentration}")
    yaxis!(yflip=true)
    p2 = plot(C[:,1] .- C[:,end], z, label=L"C(z,0) - C(z,\tau)",
    title=L"\textrm{Difference}")
    yaxis!(yflip=true)
    plot(p1, p2, layout=2)
end

# Test case 2 - Conservation of mass
test2 = false
if test2
    # Δz = 0.01, Δt = 0.1, τ = 1000, l = 100
    St = zeros(Int(trunc(τ / Δt)) + 1)
    C, t, z, K = solver_TDMA(K₁, 0, C02, St, Δt, Δz, l, τ)
    M = simps(C, z)
    plot(t,100 .*(M[1,1] .- M[:,1]) ./ M[1,1])
end

# Test case 3 - Variance increases linearly with time
test3 = false
if test3
    # Δt = 0.01, Δz = 0.01, τ = 150, l = 100.
    St = zeros(Int(trunc(τ / Δt)) + 1)
    C, t, z, K = solver_TDMA(K₂, 0, C02, St, Δt, Δz, l, τ)

    # To make the theoretical predicted curve
    K₀ = 10
    σ²₀ = 1

    # Computing the variance and expectation value from the simulation
    Cz = Array{Float64, 2}(undef, length(C[:,1]), length(C[1,:]))
    for i in 1:1:length(Cz[1,:])
        Cz[:,i] = z .* C[:,i]
    end
    μ = (simps(Cz, z) ./ simps(C, z))

    Czμ = Array{Float64, 2}(undef, length(C[:,1]), length(C[1,:]))
    for i in 1:1:length(Czμ[1,:])
        Czμ[:,i] = C[:,i] .* (z .- μ[i]).^2
    end
    σ² = (simps(Czμ, z) ./ simps(C, z))
    plot(t, σ²)
    plot!(t, σ²₀ .+ 2 .* K₀ .* t)

    # Making the theoretical predicted curve for uniform distribution
    σ²₁ = 100^2 / 12
    plot!(t, σ²₁ .* ones(length(t)))

end

# Test case 4 - Rate of mass transfer
test4 = false
if test4
    # Δt = 60 * 60, Δz = 0.01, l = 100, τ = 60 * 60 * 24 * 1000
    kw = 0.00001

    St = zeros(Int(trunc(τ / Δt)) + 1)
    C, t, z, K = solver_TDMA(K₁, kw, C03, St, Δt, Δz, l, τ)

    Mₛ = simps(C, z)
    M₀ = Mₛ[1]

    plot(t, Mₛ)

    # Plot theoretical prediction
    Mₜ = M₀ .* exp.(- t .* (kw / l))

    plot!(t, Mₜ)
end

# Teste case 5 - Equilibrium concentration
test5 = false
if test5
    # Δt = 60 * 60, Δz = 0.01, τ = 60 * 60 * 24 * 500, l = 100, K₀ = 1e-3
    kw = 0.0001

    St = Ceq .* ones(Int(trunc(τ / Δt)) + 1)
    C, t, z, K = solver_TDMA(K₁, kw, C03, St, Δt, Δz, l, τ)  # with C₀ = 0 in C03
    Cₘₐₓ = maximum(C, dims=1)[1,:]
    Cₘᵢₙ = minimum(C, dims=1)[1,:]

    plot(t, Cₘₐₓ)
    plot!(t, Cₘᵢₙ)
end

# Problem 2 - Response to changing CO₂ concentration in shallow areas

# Task 1 - Convergence test
prob2task1 = false
if prob2task1
    # l = 100, Δz = 0.01, τ = 180 days, Δt = 1 hour
    pCO2 = 415e-6       # [atm]
    Ceq = H * pCO2
    kw = a * u^2

    display("kw is "*string(kw))
    display("Ceq is "*string(Ceq))

    St = Ceq .* ones(Int(trunc(τ / Δt)) + 1)

    # Convergence tests in Δz and Δt
    point1 = false
    if point1
        timesteps = [0.1^i for i in 1:1:7]
        Δts_t, μt_error, RMSt_error = estimate_error_time_sw(kw, Ceq, timesteps, Δz, l, τ)
        plot_error_for_Δt(Δts_t, μt_error, RMSt_error, "ConvergenceInDtSW.pdf", 1e-11, 1e-11)

        spacesteps = [0.01 0.001 0.002 0.004 0.0001 0.0002 0.0004 0.00001 0.000001]
        Δzs_z, μz_error, RMSz_error = estimate_error_space_sw(kw, Ceq, spacesteps, Δt, l, τ)
        plot_error_for_Δz(Δzs_z, μz_error, RMSz_error, "ConvergenceInDzSW.pdf", 1, 1)
    end

    point2 = false
    if point2
            C, t, z, K = solver_TDMA(K₃, kw, C03, St, Δt, Δz, l, τ) # C₀ = 0 in C03
    end
end

# Task 2 - Plotting highest and lowest concentration
prob2task2 = false
if prob2task2
    # l = 100, Δz = 0.01, τ = 180 days, Δt = 1 minute
    pCO2 = 415e-6       # [atm]
    Ceq = H * pCO2
    kw = a * u^2

    display("kw is "*string(kw))
    display("Ceq is "*string(Ceq))

    St = Ceq .* ones(Int(trunc(τ / Δt)) + 1)
    C, t, z, K = solver_TDMA(K₃, kw, C03, St, Δt, Δz, l, τ) # C₀ = 0 in C03

    Cₘₐₓ = maximum(C, dims=1)[1,:]
    Cₘᵢₙ = minimum(C, dims=1)[1,:]
    plot(t, Cₘₐₓ, label=L"C_{max}(t)")
    plot!(t, Cₘᵢₙ, label=L"C_{min}(t)")
    savefig("test.pdf")
end

# Task 3 - Concentration at different times
prob2task3 = false
if prob2task3
    # l = 100, Δz = 0.01, τ = 180 days, Δt = 1 hour
    pCO2 = 415e-6       # [atm]
    Ceq = H * pCO2
    kw = a * u^2

    display("kw is "*string(kw))
    display("Ceq is "*string(Ceq))

    St = Ceq .* ones(Int(trunc(τ / Δt)) + 1)
    C, t, z, K = solver_TDMA(K₃, kw, C03, St, Δt, Δz, l, τ) # C₀ = 0 in C03

    p1 = plot(C[:,24*7], z, title=L"\textrm{After a week}",
    xlims=(0,3))
    yaxis!(yflip=true)
    Nt = length(C[1,:])
    p2 = plot(C[:,24*30], z, title=L"\textrm{After a month}",
    xlims=(0,3))
    yaxis!(yflip=true)
    p3 = plot(C[:,Int(trunc(Nt/2))], z, title=L"\textrm{After 90 days}",
    xlims=(0,3))
    yaxis!(yflip=true)
    p4 = plot(C[:,end], z, title=L"\textrm{After 180 days}",
    xlims=(0,3))
    yaxis!(yflip=true)

    plot(p1, p2, p3, p4, layout=4)
end


# Problem 4

# Task 1 - Convergence tests
prob3task1 = false
if prob3task1
    # l = 4000, τ = 1 year, Δz = 0.4, Δt = 1 hour
    pCO2 = 415e-6       # [atm]
    Ceq = H * pCO2
    kw = a * u^2
    ΔpCO2 = 2.3e-6 / (60 * 60 * 24 * 365)
    ΔCeq = H * ΔpCO2

    display("kw is "*string(kw))
    display("Ceq is "*string(Ceq))

    St = Ceq .* ones(Int(trunc(τ / Δt)) + 1)
    St = St .+ ΔCeq .* LinRange(0, τ, Int(trunc(τ / Δt)) + 1)

    C, t, z, K = solver_TDMA(K₄, kw, C03, St, Δt, Δz, l, τ) # C₀ = Ceq

    spacesteps = [0.01 0.001 0.002 0.004 0.0001 0.0002 0.0004 0.00001 0.00002 0.00004 0.000001]
    timesteps = [0.01 0.001 0.002 0.004 0.0001 0.0002 0.0004 0.00001 0.00002 0.00004 0.000001 0.0000001]
    Δts_t, μt_error, RMSt_error = estimate_error_time_dw(kw, Ceq, ΔCeq, timesteps, Δz, l, τ)
    plot_error_for_Δt(Δts_t, μt_error, RMSt_error, "ConvergenceInDtDW.pdf", 1e-15, 1e-4)

    Δzs_z, μz_error, RMSz_error = estimate_error_space_dw( kw, Ceq, ΔCeq, spacesteps, Δt, l, τ)
    plot_error_for_Δz(Δzs_z, μz_error, RMSz_error, "ConvergenceInDzDW.pdf", 1e-2, 1e-5)
end

# Task 2 - Plotting parts of the simulation
prob3task2 = false
if prob3task2
    # l = 4000, τ = 1 year, Δz = 0.4, Δt = 1 hour
    pCO2 = 415e-6       # [atm]
    Ceq = H * pCO2
    kw = a * u^2
    ΔpCO2 = 2.3e-6 / (60 * 60 * 24 * 365)
    ΔCeq = H * ΔpCO2

    display("kw is "*string(kw))
    display("Ceq is "*string(Ceq))

    St = Ceq .* ones(Int(trunc(τ / Δt)) + 1)
    St = St .+ ΔCeq .* LinRange(0, τ, Int(trunc(τ / Δt)) + 1)

    C, t, z, K = solver_TDMA(K₄, kw, C03, St, Δt, Δz, l, τ) # C₀ = Ceq
    Nt = length(C[1,:])

    plot(C[:,1], z, xlims=(2,2.3), label=L"2020", legend = :bottomright)
    plot!(C[:,Int(trunc(Nt/4))], z, label=L"\textrm{June, } 2022")
    plot!(C[:,Int(trunc(Nt/2))], z, label=L"2025")
    plot!(C[:,end], z, label=L"2030")
    yaxis!(yflip=true)
end

# Task 3 - Total mass of carbon as a function of time
prob3task3 = false
if prob3task3
    # l = 4000, τ = 1 year, Δz = 0.4, Δt = 1 hour
    pCO2 = 415e-6       # [atm]
    Ceq = H * pCO2
    kw = a * u^2
    ΔpCO2 = 2.3e-6 / (60 * 60 * 24 * 365)
    ΔCeq = H * ΔpCO2
    μ = 12e-3           # molar mass of carbon [kg]
    A = 360e12          # Total area of the oceans [m²]


    display("kw is "*string(kw))
    display("Ceq is "*string(Ceq))

    St = Ceq .* ones(Int(trunc(τ / Δt)) + 1)
    St = St .+ ΔCeq .* LinRange(0, τ, Int(trunc(τ / Δt)) + 1)

    C, t, z, K = solver_TDMA(K₄, kw, C03, St, Δt, Δz, l, τ) # C₀ = Ceq
    M = simps(C, z) .* μ .* A

    plot(t, M, legend=:bottomright, label=L"M(t)")
end

# Task 4 - Amount of absorbed carbon per year in a water column
prob3task4 = false
if prob3task4
    # l = 4000, τ = 1 year, Δz = 0.4, Δt = 1 hour
    pCO2 = 415e-6       # [atm]
    Ceq = H * pCO2
    kw = a * u^2
    ΔpCO2 = 2.3e-6 / (60 * 60 * 24 * 365)
    ΔCeq = H * ΔpCO2
    μ = 12e-3           # molar mass of carbon [kg]
    A = 360e12          # Total area of the oceans [m²]


    display("kw is "*string(kw))
    display("Ceq is "*string(Ceq))

    St = Ceq .* ones(Int(trunc(τ / Δt)) + 1)
    St = St .+ ΔCeq .* LinRange(0, τ, Int(trunc(τ / Δt)) + 1)

    C, t, z, K = solver_TDMA(K₄, kw, C03, St, Δt, Δz, l, τ) # C₀ = Ceq
    M = simps(C, z) .* μ .* 1e3
    Absorbed_C = (M[end] - M[1])/10
    display("The average amount of carbon absorbed per year: "*string(Absorbed_C)*" grams/m²")
end
  
