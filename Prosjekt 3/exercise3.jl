using Plots, LinearAlgebra, LaTeXStrings, SparseArrays, ProgressMeter

# Physical Constants

H = 5060            # Concentration proportionality constant [mol m^-3 atm^-1]
a = 6.97e-7         # Mass transfer windspeed coefficient [s/m]
pCO2 = 415e-6       # Partial pressure of CO2 [atm]
u = 10              # Constant average windspeed [m/s]
l = 100             # Depth of the ocean [m]

# Numerical constants

Δt = 0.01           # Time step [s]
Δz = 0.01           # Spatial step [m]
τ = 100             # Total time of the simulation [s]

# Relations

Ceq = H * pCO2
kw = a * u^2



# Functions

# Initializing the matrices in the simulation
function initialize_L(α, Γ, K, Kᴬ)
    du = -α / 4 .* Kᴬ[1: end-1] .- α .* K[1: end-1]
    dl = circshift(α / 4 .* Kᴬ[1 : end-1] .- α .* K[1 : end-1], -1)
    d = 1 .+ 2 * α .* K
    du[1] = -2 * α * K[1]
    dl[end] = -2 * α * K[end]
    d[1] += Γ

    return Tridiagonal(dl, d, du)
end

function initialize_R(α, Γ, K, Kᴬ)
    du = α / 4 .* Kᴬ[1 : end-1] .+ α .* K[1 : end-1]
    dl = circshift(-α / 4 .* Kᴬ[1 : end-1] .+ α .* K[1 : end-1], -1)
    d = 1 .- 2 * α .* K
    du[1] = 2 * α * K[1]
    dl[end] = 2 * α * K[end]
    d[1] -= Γ

    return Tridiagonal(dl, d, du)
end

# This function iterates forward in time using the Crank-Nicolson Method.
function time_iteration(F, R, Cⁱ, S, i)
    V = R * Cⁱ .+ (1/2) .*(S[:,i] .+ S[:,i + 1])
    return F \ V
end

# Functions for K(z)
function K₁(z)
    K₀ = 1
    return K₀ .* exp.(-z / l)
end

function K₂(z)
    K₀ = 1
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
    return exp.(.-(z .- l/2).^2 ./2)
end


# This function solves the problem.
function solver(K_func, kw, C₀, St, Δt, Δz, l, τ)
    Nt = Int(trunc(τ / Δt))
    Nz = Int(trunc(l / Δz))
    z = LinRange(0,l,Nz)
    t = LinRange(0,τ,Nt)

    K = K_func(z)
    C = Array{Float64,2}(undef, Nz, Nt)
    C[:,1] = C₀(z)
    S = spzeros(Nz,Nt)
    S[1,:] = St

    α = Δt / (2 * Δz^2)
    Γ = 2 * α * kw * Δz * (1 - (-(3/2)*K[1] + 2 * K[2] - (1/2) * K[3])/(2 * K[1]))
    Kᴬ = circshift(K, 1) .- circshift(K, -1)

    L = initialize_L(α, Γ, K, Kᴬ)
    R = initialize_R(α, Γ, K, Kᴬ)
    S = 2 .* Γ .* S
    F = lu(L)

    @showprogress 1 "Computing..." for i in 1:1:(Nt-1)
        C[:,i + 1] = time_iteration(F, R, C[:,i], S, i)
    end
    return C, t, z, K
end

# Plotting functions
function plot_mass(C, z, t)
    Δz = z[2] - z[1]
    M = sum(C, dims=1) .* Δz
    plot(t, M[1,:], label=L"M(t)",
    title=L"\textrm{Total mass as a function of time}")
    xlabel!(L"t \; [\textrm{s}]")
    ylabel!(L"M \; [\textrm{kg}]")
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
    gif(anim, "DistributionAnimation.gif", fps=30)
end

# Running tht code
St = zeros(Int(trunc(τ / Δt)))

C, t, z, K = solver(K₂, 0, C02, St, Δt, Δz, l, τ)
