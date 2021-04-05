using Plots

function Heun_solver(f, h, T, t0, y0)
    N = Int(T / h)
    y = Array{Float64, 1}(undef, N)
    t = LinRange(t0, t0 + T, N)
    y[1] = y0

    for i in 2:1:N
        yjp = y[i - 1] + h * f(t[i-1], y[i-1])
        y[i] = y[i-1] + (h/2)*(f(t[i-1],y[i-1]) + f(t[i], yjp))
    end
    return y, t
end

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


# Differential equation with known solution
function harmonic_oscillator_equation(ti, yi)
    dyi = Array{Float64, 1}(undef, length(yi))
    α = 0.5
    dyi[2] = -α^2 * yi[1]
    dyi[1] = yi[2]
    return dyi
end
