using LinearAlgebra, DataStructures, Distributions, Plots, Statistics, DelimitedFiles
using Distributed, SharedArrays

# Constants for the project

ξ = 1.0         # restitution coefficient
N = 1000       # number of particles
δt = 1/30       # time-step interval
T = 4           # time of the simulation
v0max = 10      # initial max-velocity

m = [10 1;]
r = [0.001 1;]

# Some useful functions


# This function takes in the number of particles in the system, the masses of
# the particles, the radii of the particles and the maximum allowed initial
# velocity of the particles. Returns a Nx7 array for all the particles in the
# system.
function initialize_particles(N, masses, radii, v0max)
    Ψ = Array{Float64, 2}(undef, N, 7)
    Ψ[:,1] = rand(Uniform(0.1,0.9), N)
    Ψ[:,2] = rand(Uniform(0.1,0.9), N)
    θ = rand(Uniform(0, 2 * π), N)
    Ψ[:,3] = v0max * cos.(θ)
    Ψ[:,4] = v0max * sin.(θ)
    Ψ[:,7] = zeros(N)

    n = 1
    for i in 1:1:(length(masses[:,1]))
        Ψ[n:(n + Int(N * masses[i,2]) - 1),6] .= masses[i,1]
        n += Int(N * masses[i,2])
    end
    n = 1
    for i in 1:1:length(radii[:,1])
        Ψ[n:(n + Int(N * radii[i,2]) - 1),5] .= radii[i,1]
        n += floor(N * radii[i,2])
    end
    return Ψ
end

function initialize_two_colliding_particles(masses, radii, v0max)
    Ψ = Array{Float64, 2}(undef, 2, 7)
    Ψ[1,1:2] = [0.2 0.5]
    Ψ[2,1:2] = [0.8 0.5]
    Ψ[1,3:4] = [v0max 0]
    Ψ[2,3:4] = [-v0max 0]
    Ψ[:,7] = zeros(2)

    n = 1
    for i in 1:1:(length(masses[:,1]))
        Ψ[n:(n + Int(2 * masses[i,2]) - 1),6] .= masses[i,1]
        n += Int(2 * masses[i,2])
    end
    n = 1
    for i in 1:1:length(radii[:,1])
        Ψ[n:(n + Int(2 * radii[i,2]) - 1),5] .= radii[i,1]
        n += floor(2 * radii[i,2])
    end
    display(Ψ)
    return Ψ
end

# This function takes in an array with all the particle information, the
# initialized priority queue and the number of particles. It returns all the
# initial collisiontimes.

# Must be speed up!
function initial_collisiontimes(Ψ, pq, N)
    for i in 1:1:N
        Δt_walls, nt = time_to_collision_with_walls(Ψ[i,:])

        if Δt_walls[1] != Inf
            pq[[Δt_walls[1], i, 0, nt, 0, 1]] = Δt_walls[1]
        end
        if Δt_walls[2] != Inf
            pq[[Δt_walls[2], i, 0, nt, 0, 2]] = Δt_walls[2]
        end
        for j in (i + 1):1:N
            Δt, nti, ntj = time_to_collision_with_particle(Ψ[i,:], Ψ[j,:])
            if Δt != Inf
                pq[[Δt, i, j, nti, ntj, 3]] = Δt
            end
        end
    end
    return pq
end

function initial_collisiontimes_v2(Ψ, pq, N)
    pqi = SharedArray{Float64, 3}(N, N + 1, 6)
    @time @distributed for i in 1:1:N
        Δt_walls, nt = time_to_collision_with_walls(Ψ[i,:])

        pqi[i,1,:] = [Δt_walls[1], i, 0, nt, 0, 1]
        pqi[i,2,:] = [Δt_walls[2], i, 0, nt, 0, 2]

        @distributed for j in (i + 1):1:N
            Δt, nti, ntj = time_to_collision_with_particle(Ψ[i,:], Ψ[j,:])

            pqi[i,j,:] = [Δt, i, j, nti, ntj, 3]
        end
    end
    for i in 1:1:N
        for j in 1:1:(N+1)
            if pqi[i,j,1] != Inf && pqi[i,j,2] == i
                pq[pqi[i,j,:]] = pqi[i,j,1]
            end
        end
    end
    return pq
end

# This function takes in an array with all information about a particle i
# and computes the nearest time in which it will collide with one
# of the walls. The array is ordered such that the [xx, xy, vx, vy, r, m, nt].
# Returnes array with the times until it collides with the walls and the
# number of collisions the particle has had until this point.
function time_to_collision_with_walls(Ψᵢ)
    Δt = Array{Float64, 1}(undef, 2)

    for i in 1:1:2
        if Ψᵢ[2 + i] > 0
            Δt[i] = (1 - Ψᵢ[5] - Ψᵢ[i]) / Ψᵢ[2 + i]
        elseif Ψᵢ[2 + i] < 0
            Δt[i] = (Ψᵢ[5] - Ψᵢ[i]) / Ψᵢ[2 + i]
        else
            Δt[i] = Inf
        end
    end

    return Δt, Ψᵢ[7]
end


# This function takes in an array containing all information about particle i,
# the restitution coefficient, a variable indication what wall the particle
# is colliding (1 for vertical and 2 for horizontal), the time from when the
# collision was evaluated and the number of collisions the particle had when
# this collision was listed. Returns a new array with updated particle
# information.
function collision_with_wall(Ψᵢ, ξ, vh, nt)
    if nt != Ψᵢ[7]
        return Ψᵢ
    else
        if vh == 1
            Ψᵢ[3:4] = [- ξ * Ψᵢ[3], ξ * Ψᵢ[4]]
        else
            Ψᵢ[3:4] = [ξ * Ψᵢ[3], - ξ * Ψᵢ[4]]
        end
        Ψᵢ[7] += 1

        return Ψᵢ
    end
end


# This function takes in arrays for the two particles interest. Returns the
# time until it they collide and the number of collisions both of the particles
# have had up to this point.
function time_to_collision_with_particle(Ψᵢ, Ψⱼ)
    Δx = vec(Ψⱼ[1:2] .- Ψᵢ[1:2])
    Δv = vec(Ψⱼ[3:4] .- Ψᵢ[3:4])
    Rij = Ψⱼ[5] + Ψᵢ[5]
    d = (Δv ⋅ Δx)^2 - (Δv ⋅ Δv) * (Δx ⋅ Δx - Rij^2)

    Δt = 0

    if Δv ⋅ Δx >= 0 || d <= 0 || Rij > Δx ⋅ Δx
        Δt = Inf
    else
        Δt = - (Δv ⋅ Δx + √(d)) / (Δv ⋅ Δv)
    end

    return Δt, Ψᵢ[7], Ψⱼ[7]
end


# This function takes in arrays with information about the two particles which
# are colliding, the restitution coefficient, the time since the collision was
# listed and the number of the collision of the collisions the two particles
# had when the collision was listed.
function collision_with_particle(Ψᵢ, Ψⱼ, ξ, nti, ntj)
    if Ψᵢ[7] != nti || Ψⱼ[7] != ntj
        return Ψᵢ, Ψⱼ
    else
        Δx = Ψⱼ[1:2] .- Ψᵢ[1:2]
        Δv = Ψⱼ[3:4] .- Ψᵢ[3:4]
        Rij = Ψᵢ[5] + Ψⱼ[5]

        Ψᵢ[3:4] = Ψᵢ[3:4] .+ ((1 + ξ) * (Ψⱼ[6] / (Ψᵢ[6] + Ψⱼ[6])) * ((Δv ⋅ Δx)/ Rij^2)) * Δx
        Ψⱼ[3:4] = Ψⱼ[3:4] .- ((1 + ξ) * (Ψᵢ[6] / (Ψᵢ[6] + Ψⱼ[6])) * ((Δv ⋅ Δx)/ Rij^2)) * Δx

        Ψᵢ[7] += 1
        Ψⱼ[7] += 1

        return Ψᵢ, Ψⱼ
    end
end


# This function must be speed up!
function new_times_to_collision(Ψ, event, pq)
    if event[6] != 3
        Δt_walls, nt = time_to_collision_with_walls(Ψ[Int(event[2]),:])
        if Δt_walls[1] != Inf
            pq[[event[1] + Δt_walls[1], Int(event[2]), 0, nt, 0, 1]] = event[1] + Δt_walls[1]
        end
        if Δt_walls[2] != Inf
            pq[[event[1] + Δt_walls[2], Int(event[2]), 0, nt, 0, 2]] = event[1] + Δt_walls[2]
        end
        for j in 1:1:N
            if j != Int(event[2])
                Δt, nti, ntj = time_to_collision_with_particle(Ψ[Int(event[2]),:], Ψ[j,:])
                if Δt != Inf
                    pq[[event[1] + Δt, Int(event[2]), j, nti, ntj, 3]] = event[1] + Δt
                end
            end
        end
    else
        for i in 1:1:2
            Δt_walls, nt = time_to_collision_with_walls(Ψ[Int(event[1 + i]),:])
            if Δt_walls[1] != Inf
                pq[[event[1] + Δt_walls[1], Int(event[1 + i]), 0, nt, 0, 1]] = event[1] + Δt_walls[1]
            end
            if Δt_walls[2] != Inf
                pq[[event[1] + Δt_walls[2], Int(event[1 + i]), 0, nt, 0, 2]] = event[1] + Δt_walls[2]
            end
            for j in 1:1:N
                if j != Int(event[1 + i])
                    Δt, nti, ntj = time_to_collision_with_particle(Ψ[Int(event[1 + i]),:], Ψ[j,:])
                    if Δt != Inf
                        pq[[event[1] + Δt, Int(event[1 + i]), j, nti, ntj, 3]] = event[1] + Δt
                    end
                end
            end
        end
    end
    return pq
end


function new_times_to_collision_v2(Ψ, event, pq)
    if event[6] != 3
        Δt_walls, nt = time_to_collision_with_walls(Ψ[Int(event[2]),:])
        if Δt_walls[1] != Inf
            pq[[event[1] + Δt_walls[1], Int(event[2]), 0, nt, 0, 1]] = event[1] + Δt_walls[1]
        end
        if Δt_walls[2] != Inf
            pq[[event[1] + Δt_walls[2], Int(event[2]), 0, nt, 0, 2]] = event[1] + Δt_walls[2]
        end
        pqi = SharedArray{Float64, 2}(N, 6)
        @distributed for j in 1:1:N
            if j != Int(event[2])
                Δt, nti, ntj = time_to_collision_with_particle(Ψ[Int(event[2]),:], Ψ[j,:])
                pqi[j,:] = [event[1] + Δt, Int(event[2]), j, nti, ntj, 3]
            end
        end
        for j in 1:1:N
            if pqi[j,1] != Inf && Int(event[2]) != j && pqi[j,2] != 0
                pq[pqi[j,:]] = pqi[j,1]
            end
        end
    else
        for i in 1:1:2
            Δt_walls, nt = time_to_collision_with_walls(Ψ[Int(event[1 + i]),:])
            if Δt_walls[1] != Inf
                pq[[event[1] + Δt_walls[1], Int(event[1 + i]), 0, nt, 0, 1]] = event[1] + Δt_walls[1]
            end
            if Δt_walls[2] != Inf
                pq[[event[1] + Δt_walls[2], Int(event[1 + i]), 0, nt, 0, 2]] = event[1] + Δt_walls[2]
            end
            pqi = SharedArray{Float64, 2}(N, 6)
            @distributed for j in 1:1:N
                if j != Int(event[1 + i])
                    Δt, nti, ntj = time_to_collision_with_particle(Ψ[Int(event[1 + i]),:], Ψ[j,:])
                    pqi[j,:] = [event[1] + Δt, Int(event[1 + i]), j, nti, ntj, 3]
                end
            end
        end
        for j in 1:1:N
            if pqi[j,1] != Inf && Int(event[2]) != j && pqi[j,2] != 0
                pq[pqi[j,:]] = pqi[j,1]
            end
        end
    end
    return pq
end



function main(N, ξ, masses, radii, v0max, T, δt)
    Ψ = initialize_particles(N, masses, radii, v0max)
    particle_tracks = Array{Float64, 2}(undef, N, 2)
    particle_tracks[:,1:2] .= Ψ[:,1:2]
    velocities = Array{Float64, 1}(undef, N)
    velocities[:] = Ψ[:,3].^2 .+ Ψ[:,4].^2

    t = 0

    pq = PriorityQueue{Array{Float64, 1}, Float64}()
    pq = initial_collisiontimes(Ψ, pq, N)
    display(pq)

    # Array t, i, j, nti, ntj, wp (vertical = 1, horisontal = 2, particle = 3)
    event = dequeue!(pq)

    ϵ = 0
    while t < T
        tt = t
        while t + δt > event[1] && t < event[1]
            Δt = event[1] - tt
            Ψ[:,1:2] = Ψ[:,1:2] .+ Ψ[:,3:4] * Δt

            if event[6] == 1 || event[6] == 2
                Ψ[Int(event[2]),:] = collision_with_wall(Ψ[Int(event[2]),:], ξ, event[6], event[4])
                pq = new_times_to_collision(Ψ, event, pq)
            else
                Ψᵢ = Ψ[Int(event[2]),:]
                Ψⱼ = Ψ[Int(event[3]),:]
                Ψᵢ, Ψⱼ = collision_with_particle(Ψᵢ, Ψⱼ, ξ, event[4], event[5])
                Ψ[Int(event[2]),:] = Ψᵢ
                Ψ[Int(event[3]),:] = Ψⱼ
                pq = new_times_to_collision(Ψ, event, pq)
            end

            tt = event[1]
            event = dequeue!(pq)
        end
        Ψ[:,1:2] = Ψ[:,1:2] .+ Ψ[:,3:4] * (t + δt - tt)
        particle_tracks = hcat(particle_tracks, Ψ[:,1:2])
        velocities = hcat(velocities, Ψ[:,3].^2 .+ Ψ[:,4].^2)
        t += δt
        ϵ += 1
    end
    display("returned")
    return particle_tracks, velocities
end


function problem_1(N, ξ, masses, radii, v0max, it)
    Ψ = initialize_particles(N, masses, radii, v0max)
    #velocities = Array{Float64, 2}(undef, N, it + 1)
    #velocities[:,1] = sqrt.(Ψ[:,3].^2 .+ Ψ[:,4].^2)
    #times = Array{Float64, 1}(undef, it + 1)
    #times[1] = 0

    t = 0
    k = 1

    velocities = Array{Float64, 2}(undef, N, 4)

    pq = PriorityQueue{Array{Float64, 1}, Float64}()
    pq = initial_collisiontimes(Ψ, pq, N)

    event = dequeue!(pq)
    coll_count = 0
    collww = 0
    collwp = 0
    display("Running...")

    for i in 2:(it + 1)
        Δt = event[1] - t
        Ψ[:,1:2] = Ψ[:,1:2] .+ Ψ[:,3:4] * Δt

        if event[6] == 1 || event[6] == 2
            collww += 1
            Ψ[Int(event[2]),:] = collision_with_wall(Ψ[Int(event[2]),:], ξ, event[6], event[4])
            pq = new_times_to_collision(Ψ, event, pq)
        else
            collwp += 1
            Ψᵢ = Ψ[Int(event[2]),:]
            Ψⱼ = Ψ[Int(event[3]),:]
            Ψᵢ, Ψⱼ = collision_with_particle(Ψᵢ, Ψⱼ, ξ, event[4], event[5])
            Ψ[Int(event[2]),:] = Ψᵢ
            Ψ[Int(event[3]),:] = Ψⱼ
            pq = new_times_to_collision(Ψ, event, pq)
        end
        #velocities[:,i] = sqrt.(Ψ[:,3].^2 .+ Ψ[:,4].^2)
        #times[i] = t

        t = event[1]
        event = dequeue!(pq)
        coll_count += 1

        if coll_count == 700000 || coll_count == 800000 || coll_count == 900000
            velocities[:,k] = sqrt.(Ψ[:,3].^2 .+ Ψ[:,4].^2)
            k += 1
        end

        if coll_count % (it//500) == 0
            display(coll_count * 100 / it)
        end
    end
    display("Returned")
    display(collww)
    display(collwp)

    velocities[:,k] = sqrt.(Ψ[:,3].^2 .+ Ψ[:,4].^2)

    return velocities
end

@time problem_1(N, ξ, m, r, v0max, 1000)


#vel = problem_1(N, ξ, m, r, v0max, 1000000)

#writedlm("velocities15.csv", vel, ',')

#vel = problem_1(N, ξ, m, r, v0max, 1000000)

#writedlm("velocities16.csv", vel, ',')

#vel = problem_1(N, ξ, m, r, v0max, 1000000)

#writedlm("velocities17.csv", vel, ',')

#vel = problem_1(N, ξ, m, r, v0max, 1000000)

#writedlm("velocities18.csv", vel, ',')

#vel = problem_1(N, ξ, m, r, v0max, 1000000)

#writedlm("velocities19.csv", vel, ',')

#vel = problem_1(N, ξ, m, r, v0max, 1000000)

#writedlm("velocities20.csv", vel, ',')


#velsq = vel.^2
#E = transpose(sum(velsq, dims=1))
#avg_vel = transpose(mean(vel, dims=1))

#writedlm("times.csv", ti, ',')


#particle_x = particle_track[:,1:2:end]
#particle_y = particle_track[:,2:2:end]

#n = Int(length(particle_track[1,:]) / 2)

#gr()
#anim = @animate for i ∈ 1:n
#    plot(particle_x[:,i], particle_y[:,i], seriestype= :scatter,
#     title = "Scatter plot", xlims=(0,1), ylims=(0,1), legend = false)
#end

#gif(anim, "Scatterplot.gif", fps=30)
