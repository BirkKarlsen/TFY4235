using Plots, DelimitedFiles, LaTeXStrings

A = readdlm("velocities_parallel1.csv")

plot(A[:,1], seriestype=:histogram, bins=30, normalize=true)

function read_velocities(filenames)
    A = Array{Float64, 1}()
    for name in filenames
        A = vcat(A, readdlm(name))
    end
    return A
end

function plot_velocity_distribution_homogeneour(A)
    plot(A[:,1], seriestype=:histogram, bins=50, normalize=true)
    plot!(title = L"\textrm{Velocity distribution}")
    savefig("test.pdf")
end

fn = ["velocities_parallel"*string(i)*".csv" for i ∈ 1:1:20]

A = read_velocities(fn)
plot_velocity_distribution_homogeneour(A)
