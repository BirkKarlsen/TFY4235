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
    plot(A[:,1], seriestype=:histogram, bins=100, normalize=true,
    label=L"\textrm{Simulation}")
    plot!(v, fv, label=L"\textrm{Maxwell-Boltzmann}", xlims=(0,1.5))
    plot!(title = L"\textrm{Final speed distribution}")
    xlabel!(L"v \; [\textrm{m/s}]")
    ylabel!(L"p \; [\textrm{-}]")
    savefig("problem11.pdf")
end

function plot_initial_velocity_distribution(A)
    plot(A, seriestype=:histogram, normalize=true, bins=200,
    label=L"\textrm{Simulation}")
    plot!(xlims=(0,1.5))
    plot!(title = L"\textrm{Initial speed distribution}")
    xlabel!(L"v \; [\textrm{m/s}]")
    ylabel!(L"p \; [\textrm{-}]")
    savefig("problem12.pdf")
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

    p1 = plot(append!(v11[:,1], [-0.1, 4]), seriestype=:histogram, label=L"m = m_0", bins = 200, normalize=true,
    alpha=0.7)
    plot!(append!(v21[:,1], [-0.1, 4]), seriestype=:histogram, label=L"m = 4 m_0", bins = 200, normalize=true,
    alpha=0.7)
    ylabel!(L"p \; \; [-]")
    xlabel!(L"v \; \; [\textrm{m/s}]")
    plot!(xlims=(0,3), title=L"\textrm{Initial distribution}")

    v = LinRange(0, 3, 200)
    fv1 = Maxwell_Boltzmann_distribution(v, 6.5 / 2)
    fv2 = Maxwell_Boltzmann_distribution(v, 2 * 6.5)

    p2 = plot(v12[:,1], seriestype=:histogram, label=L"m = m_0", bins = 50, normalize=true, alpha=0.7)
    plot!(v22[:,1], seriestype=:histogram, label=L"m = 4 m_0", bins = 50, normalize=true, alpha=0.7)
    plot!(v, fv1, label=L"\textrm{Theoretical for } m=m_0")
    plot!(v, fv2, label=L"\textrm{Theoretical for } m=4m_0")
    plot!(xlims=(0,3), title=L"\textrm{Final distribution}")
    ylabel!(L"p \; \; [-]")
    xlabel!(L"v \; \; [\textrm{m/s}]")
    plot(p1, p2, layout=(2,1))
    savefig("problem21.pdf")

    p1 = plot([ti ti], [av1 av2], title=L"\textrm{Average speed}",
    label=[L"m = m_0" L"m = 4 m_0"], legend = :right)
    xlabel!(L"t \; \; [\textrm{s}]")
    ylabel!(L"v \; \; [\textrm{m/s}]")

    p2 = plot([ti ti], [k1 k2], title=L"\textrm{Average kinetic energy}",
    label=[L"m = m_0" L"m = 4 m_0"])
    xlabel!(L"t \; \; [\textrm{s}]")
    ylabel!(L"K \; \; [\textrm{J}]")
    plot(p1, p2, layout=(2,1))
    savefig("problem22.pdf")
end


# For problem 1
pltprob1 = false
if pltprob1
    fn = ["final_velocities_parallel"*string(i)*".csv" for i ∈ 1:1:20]
    A = read_velocities(fn)
    plot_velocity_distribution_homogeneous(A)

    fn = ["initial_velocities_parallel"*string(i)*".csv" for i ∈ 1:1:20]
    A = read_velocities(fn)[:,1]
    append!(A, [-0.1, 2.0])
    plot_initial_velocity_distribution(A)
end

# For problem 2
pltprob2 = false
if pltprob2
    plot_for_problem_2()
end

# For problem 4:
pltprob4 = false
if pltprob4
    cv = readdlm("craters_v.csv")
    vv = readdlm("craters_vv.csv")

    scatter(vv, cv, shape=:x)
    savefig("problem41.pdf")
end


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
