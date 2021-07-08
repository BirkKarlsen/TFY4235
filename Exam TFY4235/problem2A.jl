using LinearAlgebra, Plots, ProgressMeter, LaTeXStrings

# Functions for Problem 2A

# Functions for the model

# This function is a general solver for a set of coupled differential
# equations. Here it is assumed that y for each timestep is  a vector of
# functions which change with time:
function solver(f, alg, params, h, T, t0, y0)
    N = Int(trunc(T/h))
    t = LinRange(t0, T + t0, N)
    y = Array{Float64, 2}(undef, length(y0), N)
    y[:,1] = y0
    @showprogress 1 "Computing..." for i in 2:1:N
        y[:,i] = alg(f, h, y[:,i-1], t[i-1], params)
    end
    return y, t
end


function optimize_β(params, h, T, y0, Nβ)
    βs = LinRange(0,1,Nβ)
    I_max = Array{Float64, 1}(undef, Nβ)
    for i in 1:1:Nβ
        params[1] = βs[i]
        yi, ti = solver(deterministic_SIR, RK4_algorithm, params, h, T, 0, y0)
        I_max[i] = maximum(yi[2,:])
    end
    return βs, I_max
end

function optimize_vacc(params, h, T, y0, Nvacc)
    vaccs = LinRange(0, 1 - y0[2], Nvacc)
    I_max = Array{Float64, 1}(undef, Nvacc)
    for i in 1:1:Nvacc
        y0[1] = 1 - y0[2] - vaccs[i]
        y0[3] = vaccs[i]
        yi, ti = solver(deterministic_SIR, RK4_algorithm, params, h, T, 0, y0)
        I_max[i] = maximum(yi[2,:])
    end
    return vaccs, I_max
end

function bisection_method_β(target, βa, βb, f, alg, params, h, T, t0, y0, max_it, tol)
    params[1] = βa
    ya, t = solver(f, alg, params, h, T, t0, y0)
    Imaxa = maximum(ya[2,:])

    params[1] = βb
    yb, t = solver(f, alg, params, h, T, t0, y0)
    Imaxb = maximum(yb[2,:])

    βc = (βa + βb)/2
    params[1] = βc
    yc, t = solver(f, alg, params, h, T, t0, y0)
    Imaxc = maximum(yc[2,:])

    err = target - Imaxc

    i = 0
    while abs(err) > tol
        i += 1
        if i > max_it
            display("Failed to converge")
            return βc, Imaxc
        end
        if (target - Imaxc) * (target - Imaxa) < 0
            βb = βc
            Imaxb = Imaxc
        else
            βa = βc
            Imaxa = Imaxc
        end

        βc = (βa + βb)/2
        params[1] = βc
        yc, t = solver(f, alg, params, h, T, t0, y0)
        Imaxc = maximum(yc[2,:])
        err = target - Imaxc
        display("iteration "*string(i)*", error is "*string(err))
    end
    return βc, Imaxc
end

function bisection_method_vacc(target, vacca, vaccb, f, alg, params, h, T, t0, y0, max_it, tol)
    display(y0)
    display(target)

    y0a = copy(y0)
    y0a[1] -= vacca
    y0a[3] += vacca
    ya, t = solver(f, alg, params, h, T, t0, y0a)
    Imaxa = maximum(ya[2,2:end])

    y0b = copy(y0)
    y0b[1] -= vaccb
    y0b[3] += vaccb
    yb, t = solver(f, alg, params, h, T, t0, y0b)
    Imaxb = maximum(yb[2,2:end])

    vaccc = (vacca + vaccb)/2
    y0c = copy(y0)
    y0c[1] -= vaccc
    y0c[3] += vaccc
    yc, t = solver(f, alg, params, h, T, t0, y0c)
    Imaxc = maximum(yc[2,2:end])

    err = target - Imaxc

    i = 0
    while abs(err) > tol
        i += 1
        if i > max_it
            display("Failed to converge")
            return vaccc, Imaxc
        end
        if (target - Imaxc) * (target - Imaxa) < 0
            vaccb = vaccc
            Imaxb = Imaxc
        else
            vacca = vaccc
            Imaxa = Imaxc
        end

        vaccc = (vacca + vaccb)/2
        y0c = copy(y0)
        y0c[1] -= vaccc
        y0c[3] += vaccc
        yc, t = solver(f, alg, params, h, T, t0, y0c)
        Imaxc = maximum(yc[2,2:end])
        err = target - Imaxc
        display("iteration "*string(i)*", error is "*string(err))
    end
    return vaccc, Imaxc
end

# Some essential algorithms for solving ODEs

# This is tne simple forward Euler method
function euler_algorithm(f, h, yi, ti, params)
    return yi .+ h .* f(ti, yi, params)
end

# This is the fourth order Runge-Kutta method
function RK4_algorithm(f, h, yi, ti, params)
    s₁ = f(ti, yi, params)
    s₂ = f(ti + h/2, yi .+ s₁ .* (h/2), params)
    s₃ = f(ti + h/2, yi .+ s₂ .* (h/2), params)
    s₄ = f(ti + h, yi .+ h .* s₃, params)
    return yi .+ (h/6) .* (s₁ .+ 2 .* s₂ .+ 2 .* s₃ .+ s₄)
end

# The set of equations that describe the deterministic SIR model
function deterministic_SIR(ti, yi, params)
    dy = Array{Float64, 1}(undef, length(yi))
    β = params[1]
    τ = params[2]
    N = params[3]

    dy[1] = -(β/N) * yi[1] * yi[2]              # derivative for S
    dy[2] = (β/N) * yi[1] * yi[2] - yi[2] / τ   # derivative for I
    dy[3] = yi[2] / τ                           # derivative for R
    return dy
end

# Analytic expression for S(∞) and R(∞)
function asymptotic_SR(ℛ₀, N, S₀, R₀, Nit)
    S_it = Array{Float64, 1}(undef, Nit)
    R_it = Array{Float64, 1}(undef, Nit)
    S_it_stop = Nit
    R_it_stop = Nit
    S_it[1] = S₀
    R_it[1] = R₀

    for i in 2:1:Nit
        S_it[i] = exp(-ℛ₀ * (1 - S_it[i-1]))
    end

    for i in 2:1:Nit
        R_it[i] = 1 - exp(-ℛ₀ * R_it[i-1])
    end

    return S_it[S_it_stop], R_it[R_it_stop]
end

function analytic_I(y0, t)
    return y0[2] .* exp.((params[1] - 1/params[2]) .* t)
end
