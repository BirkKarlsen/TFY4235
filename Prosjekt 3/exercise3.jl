using Plots, LinearAlgebra, LaTeXStrings, SparseArrays, ProgressMeter
using DelimitedFiles, QuadGK

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

l = 100
τ = 100
Δt = 0.1
Δz = 0.01

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

    L = initialize_L(α, Γ, K, Kᴬ)
    R = initialize_R(α, Γ, K, Kᴬ)
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
    s = s .+ 4 .* transpose(sum(f[2:2:end,:], dims=1))
    s = s .+ 2 .* transpose(sum(f[3:2:end-1,:], dims=1))
    return (Δx/3) .* s
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

function plot_variance(C, z, t)
    Δz = z[2] - z[1]
    μ = sum(C .* z, dims=1) ./ sum(C, dims=1)
    σ = sum(C .* (z .- μ).^2, dims=1) ./ sum(C, dims=1)


    plot([t t], [σ[1,:] μ[1,:]])
end

function animate_massdistribution(C, z, t, nl)
    gr()
    anim = @animate for i in 1:nl:length(t)
        plot(C[:,i], z, label=L"C(z,t)", xlims=(0,1),
        title=L"\textrm{Mass Distribution}")
        yaxis!(yflip=true)
        xlabel!(L"C(z,t)")
        ylabel!(L"z \; [\textrm{m}]")
    end
    gif(anim, "DistributionAnimation2.gif", fps=30)
end

# Functions for K(z)

function K₁(z)
    K₀ = 60
    return K₀ .* exp.(-z / l)
end

function K₂(z)
    K₀ = 5
    return K₀ .* ones(length(z))
end

function K₃(z)
    K₀ = 1e-3           # [m^2/s]
    Kₐ = 2e-2           # [m^2/s]
    zₐ = 7              # [m]
    Kᵦ = 5e-2           # [m^2/s]
    zᵦ = 10             # [m]
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
    return exp.(.-z./l)
end

function C02(z)
    return exp.(.-(z .- l/2).^2 ./(2 * 1^2))
end

function C03(z)
    C₀ = 0.1
    return C₀ .* ones(length(z))
end


# Running tht code

St = zeros(Int(trunc(τ / Δt)) + 1)

C, t, z, K = solver_TDMA(K₁, 0, C02, St, Δt, Δz, l, τ)
animate_massdistribution(C, z, t, 10)
plot_mass(C, z, t)
