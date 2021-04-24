using Plots, DelimitedFiles, LaTeXStrings


function read_velocities(filenames)
    A = Array{Float64, 1}()
    for name in filenames
        A = vcat(A, readdlm(name))
    end
    return A
end

function Maxwell_Boltzmann_distribution(v, a)
    return a .* v .* exp.(-a .* v.^2 ./ 2)
end

function plot_velocity_distribution_homogeneous(A)
    v = LinRange(0, 1.5, 200)
    fv = Maxwell_Boltzmann_distribution(v, 8)
    plot(A[:,1], seriestype=:histogram, bins=50, normalize=true,
    label=L"\textrm{Simulation}")
    plot!(v, fv, label=L"\textrm{Maxwell-Boltzmann}")
    plot!(title = L"\textrm{Velocity distribution}")
    xlabel!(L"v \; [\textrm{m/s}]")
    ylabel!(L"p \; [\textrm{-}]")
    savefig("problem11.pdf")
end


function plot_for_problem_2()

    vs = readdlm("vels1.csv")
    avs = readdlm("avg_vels1.csv")
    ks = readdlm("avg_kin1.csv")
    ti = readdlm("time1.csv")
    N1 = Int(length(vs)/2)
    N2 = Int(N1/2)
    Nt = Int(length(ti))
    v1, v2 = vs[1:N1], vs[N1 + 1:end]
    v11, v12 = v1[1:N2], v1[N2 + 1:end]
    v21, v22 = v2[1:N2], v2[N2 + 1:end]
    av1, av2 = avs[1:Nt], avs[Nt + 1:end]
    k1, k2, = ks[1:Nt], ks[Nt + 1:end]

    vs = readdlm("vels2.csv")
    v1, v2 = vs[1:N1], vs[N1 + 1:end]
    v11 = vcat(v11, v1[1:N2])
    v12 = vcat(v12, v1[N2 + 1:end])
    v21 = vcat(v21, v2[1:N2])
    v22 = vcat(v22, v2[N2 + 1:end])

    plot(v12[:,1], seriestype=:histogram, label=L"m = m_0", bins = 50, normalize=true)
    plot!(v22[:,1], seriestype=:histogram, label=L"m = 4 m_0", bins = 50, normalize=true)
    #plot([ti ti], [av1 av2])
    #plot([ti ti], [k1 k2])
end

fn = ["velocities_parallel"*string(i)*".csv" for i ∈ 1:1:20]
A = read_velocities(fn)
plot_velocity_distribution_homogeneous(A)

#vs = readdlm("vels3.csv")
#avs = readdlm("avg_vels3.csv")
#ks = readdlm("avg_kin3.csv")
#ti = readdlm("time3.csv")
#N1 = Int(length(vs)/2)
#N2 = Int(N1/2)
#Nt = Int(length(ti))
#v1, v2 = vs[1:N1], vs[N1 + 1:end]
#v11, v12 = v1[1:N2], v1[N2 + 1:end]
#v21, v22 = v2[1:N2], v2[N2 + 1:end]
#av1, av2 = avs[1:Nt], avs[Nt + 1:end]
#k1, k2, = ks[1:Nt], ks[Nt + 1:end]

#plot([ti ti], [av1 av2])
#plot([ti ti], [k1 k2])
