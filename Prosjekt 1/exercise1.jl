using LinearAlgebra, DataStructures, Distributions, Plots
using ProgressMeter, FLoops, Statistics, DelimitedFiles

# Constants for the project

ξ = 1.0         # restitution coefficient
N = 1000        # number of particles
δt = 1/30       # time-step interval
T = 10          # time of the simulation
v0max = 10/20   # initial max-velocity

rm = 0.01
rm2 = 0.05
m = [10 1;]
r = [rm 1;]


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
        Ψ[Int(n):Int(n + Int(N * radii[i,2]) - 1),5] .= radii[i,1]
        n += floor(N * radii[i,2])
    end
    return Ψ
end

function initialize_two_colliding_particles(masses, radii, v0max, b)
    Ψ = Array{Float64, 2}(undef, 2, 7)
    Ψ[1,1:2] = [0.2 0.5]
    Ψ[2,1:2] = [0.8 0.5+b]
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
        Ψ[Int(n):Int(n + Int(2 * radii[i,2]) - 1),5] .= radii[i,1]
        n += floor(2 * radii[i,2])
    end
    display(Ψ)
    return Ψ
end

# This function takes in an array with all the particle information, the
# initialized priority queue and the number of particles. It returns all the
# initial collisiontimes.

function initial_collisiontimes(Ψ, pq, N)
    for i in 1:1:N
        Δt_walls, nt = time_to_collision_with_walls(Ψ[i,:])

        if Δt_walls[1] != Inf
            pq[[Δt_walls[1], i, 0, nt, 0, 1, 0]] = Δt_walls[1]
        end
        if Δt_walls[2] != Inf
            pq[[Δt_walls[2], i, 0, nt, 0, 2, 0]] = Δt_walls[2]
        end
        for j in (i + 1):1:N
            Δt, nti, ntj = time_to_collision_with_particle(Ψ[i,:], Ψ[j,:])
            if Δt != Inf
                pq[[Δt, i, j, nti, ntj, 3, 0]] = Δt
            end
        end
    end
    return pq
end

# The same function but using FLoops
function initial_collisiontimes_parallel(Ψ, pq, N, ncores)
    @floop ThreadedEx(basesize=N+ncores) for i in 1:1:N
        Δt_walls, nt = time_to_collision_with_walls(Ψ[i,:])

        if Δt_walls[1] != Inf
            pq[[Δt_walls[1], i, 0, nt, 0, 1, 0]] = Δt_walls[1]
        end
        if Δt_walls[2] != Inf
            pq[[Δt_walls[2], i, 0, nt, 0, 2, 0]] = Δt_walls[2]
        end
        for j in (i + 1):1:N
            Δt, nti, ntj = time_to_collision_with_particle(Ψ[i,:], Ψ[j,:])
            if Δt != Inf
                pq[[Δt, i, j, nti, ntj, 3, 0]] = Δt
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
    if Int(nt) != Int(Ψᵢ[7])
        return Ψᵢ, false
    else
        if vh == 1
            Ψᵢ[3:4] = [- ξ * Ψᵢ[3], ξ * Ψᵢ[4]]
        else
            Ψᵢ[3:4] = [ξ * Ψᵢ[3], - ξ * Ψᵢ[4]]
        end
        Ψᵢ[7] += 1

        return Ψᵢ, true
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

    if Δv ⋅ Δx >= 0 || d <= 0 || Rij > √(Δx ⋅ Δx)
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
    if Int(Ψᵢ[7]) != Int(nti) || Int(Ψⱼ[7]) != Int(ntj)
        return Ψᵢ, Ψⱼ, false
    else
        Δx = Ψⱼ[1:2] .- Ψᵢ[1:2]
        Δv = Ψⱼ[3:4] .- Ψᵢ[3:4]
        Rij = Ψᵢ[5] + Ψⱼ[5]

        Ψᵢ[3:4] = Ψᵢ[3:4] .+ ((1 + ξ) * (Ψⱼ[6] / (Ψᵢ[6] + Ψⱼ[6])) * ((Δv ⋅ Δx)/ Rij^2)) * Δx
        Ψⱼ[3:4] = Ψⱼ[3:4] .- ((1 + ξ) * (Ψᵢ[6] / (Ψᵢ[6] + Ψⱼ[6])) * ((Δv ⋅ Δx)/ Rij^2)) * Δx

        Ψᵢ[7] += 1
        Ψⱼ[7] += 1

        return Ψᵢ, Ψⱼ, true
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

# Same as last function but using FLoops
function new_times_to_collision_parallel(Ψ, event, pq, c, ncores)
    if event[6] != 3
        Δt_walls, nt = time_to_collision_with_walls(Ψ[Int(event[2]),:])
        if Δt_walls[1] != Inf
            pq[[event[1] + Δt_walls[1], Int(event[2]), 0, nt, 0, 1, c]] = event[1] + Δt_walls[1]
        end
        if Δt_walls[2] != Inf
            pq[[event[1] + Δt_walls[2], Int(event[2]), 0, nt, 0, 2, c]] = event[1] + Δt_walls[2]
        end
        @floop ThreadedEx(basesize = N + ncores) for j in 1:1:N
            if j != Int(event[2])
                Δt, nti, ntj = time_to_collision_with_particle(Ψ[Int(event[2]),:], Ψ[j,:])
                if Δt != Inf
                    pq[[event[1] + Δt, Int(event[2]), j, nti, ntj, 3, c]] = event[1] + Δt
                end
            end
        end
    else
        for i in 1:1:2
            Δt_walls, nt = time_to_collision_with_walls(Ψ[Int(event[1 + i]),:])
            if Δt_walls[1] != Inf
                pq[[event[1] + Δt_walls[1], Int(event[1 + i]), 0, nt, 0, 1, c]] = event[1] + Δt_walls[1]
            end
            if Δt_walls[2] != Inf
                pq[[event[1] + Δt_walls[2], Int(event[1 + i]), 0, nt, 0, 2, c]] = event[1] + Δt_walls[2]
            end
            @floop ThreadedEx(basesize = N + ncores) for j in 1:1:N
                if j != Int(event[1 + i])
                    Δt, nti, ntj = time_to_collision_with_particle(Ψ[Int(event[1 + i]),:], Ψ[j,:])
                    if Δt != Inf
                        pq[[event[1] + Δt, Int(event[1 + i]), j, nti, ntj, 3, c]] = event[1] + Δt
                    end
                end
            end
        end
    end
    return pq
end


# This function animates a set of particle tracks modelling a homogeneous gas.
function animate_homogeneous_gas(particle_tracks, rm)
    particle_x = particle_track[:,1:2:end]
    particle_y = particle_track[:,2:2:end]

    n = Int(length(particle_track[1,:]) / 2)

    gr()
    anim = @animate for i ∈ 1:n
        plot(particle_x[1:end,i], particle_y[1:end,i], seriestype= :scatter,
         xlims=(0,1), ylims=(0,1), legend = false,
         markersize=rm*550*1.101, size=(600, 600))
    end

    gif(anim, "Scatterplotp.gif", fps=30)
end


function main(N, ξ, masses, radii, v0max, T, δt)
    Ψ = initialize_particles(N, masses, radii, v0max)
    Ψ = initialize_two_colliding_particles(masses, radii, v0max)
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

function mainp(N, ξ, masses, radii, v0max, T, δt, ncores)
    Ψ = initialize_particles(N, masses, radii, v0max)
    #Ψ = initialize_two_colliding_particles(masses, radii, v0max, 0)
    particle_tracks = Array{Float64, 2}(undef, N, 2)
    particle_tracks[:,1:2] .= Ψ[:,1:2]
    velocities = Array{Float64, 1}(undef, N)
    velocities[:] = Ψ[:,3].^2 .+ Ψ[:,4].^2

    t = 0

    pq = PriorityQueue{Array{Float64, 1}, Float64}()
    pq = initial_collisiontimes_parallel(Ψ, pq, N, ncores)

    # Array t, i, j, nti, ntj, wp (vertical = 1, horisontal = 2, particle = 3), c
    event = dequeue!(pq)
    coll = 0
    ϵ = 0
    while t < T
        tt = t
        while t + δt > event[1] && t < event[1]
            Δt = event[1] - tt
            Ψ[:,1:2] = Ψ[:,1:2] .+ Ψ[:,3:4] * Δt

            if event[6] == 1 || event[6] == 2
                Ψ[Int(event[2]),:], happen = collision_with_wall(Ψ[Int(event[2]),:], ξ, event[6], event[4])
                if happen
                    coll += 1
                    pq = new_times_to_collision_parallel(Ψ, event, pq, coll, ncores)
                end
            else
                Ψᵢ = Ψ[Int(event[2]),:]
                Ψⱼ = Ψ[Int(event[3]),:]
                Ψᵢ, Ψⱼ, happen = collision_with_particle(Ψᵢ, Ψⱼ, ξ, event[4], event[5])
                Ψ[Int(event[2]),:] = Ψᵢ
                Ψ[Int(event[3]),:] = Ψⱼ
                if happen
                    coll += 1
                    pq = new_times_to_collision_parallel(Ψ, event, pq, coll, ncores)
                end
            end

            tt = event[1]
            if isempty(pq)
                break
            end
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



# These functions are to solve Problem 1
function problem_1(N, ξ, masses, radii, v0max, it)
    Ψ = initialize_particles(N, masses, radii, v0max)

    t = 0

    velocities = Array{Float64, 1}(undef, N)

    pq = PriorityQueue{Array{Float64, 1}, Float64}()
    pq = initial_collisiontimes(Ψ, pq, N)

    event = dequeue!(pq)
    coll_count = 0
    collww = 0
    collwp = 0

    @showprogress 1 "Computing..." for i in 2:(it + 1)
        Δt = event[1] - t
        Ψ[:,1:2] = Ψ[:,1:2] .+ Ψ[:,3:4] * Δt

        if event[6] == 1 || event[6] == 2
            collww += 1
            Ψ[Int(event[2]),:], happen = collision_with_wall(Ψ[Int(event[2]),:], ξ, event[6], event[4])
            if happen
                pq = new_times_to_collision(Ψ, event, pq)
            end
        else
            collwp += 1
            Ψᵢ = Ψ[Int(event[2]),:]
            Ψⱼ = Ψ[Int(event[3]),:]
            Ψᵢ, Ψⱼ, happen = collision_with_particle(Ψᵢ, Ψⱼ, ξ, event[4], event[5])
            Ψ[Int(event[2]),:] = Ψᵢ
            Ψ[Int(event[3]),:] = Ψⱼ
            if happen
                pq = new_times_to_collision(Ψ, event, pq)
            end
        end

        t = event[1]
        event = dequeue!(pq)
        coll_count += 1
    end
    display("Returned")
    display(collww)
    display(collwp)

    velocities = sqrt.(Ψ[:,3].^2 .+ Ψ[:,4].^2)
    return velocities
end


function problem_1p(N, ξ, masses, radii, v0max, it, ncores)
    Ψ = initialize_particles(N, masses, radii, v0max)

    t = 0

    velocities = Array{Float64, 1}(undef, N)

    pq = PriorityQueue{Array{Float64, 1}, Float64}()
    pq = initial_collisiontimes_parallel(Ψ, pq, N, ncores)

    event = dequeue!(pq)
    coll_count = 0
    collww = 0
    collwp = 0

    @showprogress 1 "Computing..." for i in 2:(it + 1)
        Δt = event[1] - t
        Ψ[:,1:2] = Ψ[:,1:2] .+ Ψ[:,3:4] * Δt

        if event[6] == 1 || event[6] == 2
            collww += 1
            Ψ[Int(event[2]),:], happen = collision_with_wall(Ψ[Int(event[2]),:], ξ, event[6], event[4])
            if happen
                pq = new_times_to_collision_parallel(Ψ, event, pq, ncores)
            end
        else
            collwp += 1
            Ψᵢ = Ψ[Int(event[2]),:]
            Ψⱼ = Ψ[Int(event[3]),:]
            Ψᵢ, Ψⱼ, happen = collision_with_particle(Ψᵢ, Ψⱼ, ξ, event[4], event[5])
            Ψ[Int(event[2]),:] = Ψᵢ
            Ψ[Int(event[3]),:] = Ψⱼ
            if happen
                pq = new_times_to_collision_parallel(Ψ, event, pq, ncores)
            end
        end

        t = event[1]
        event = dequeue!(pq)
        coll_count += 1
    end
    display("Returned")
    display(collww)
    display(collwp)

    velocities = sqrt.(Ψ[:,3].^2 .+ Ψ[:,4].^2)
    return velocities
end

function generate_statistics(it, n, ncores)
    for i ∈ 1:1:n
        vel = problem_1p(N, ξ, m, r, v0max, it, ncores)
        filename = "velocities_parallel" * string(i) * ".csv"
        writedlm(filename, vel, ',')
        display(filename)
    end
end


#velsq = vel.^2
#E = transpose(sum(velsq, dims=1))
#avg_vel = transpose(mean(vel, dims=1))

#writedlm("times.csv", ti, ',')

#particle_track, velocities = mainp(N, ξ, m, r, v0max, T, δt, ncores)
