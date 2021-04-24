using LinearAlgebra, DataStructures, Distributions, Plots
using ProgressMeter, FLoops, Statistics, DelimitedFiles
using LaTeXStrings

# Constants for the project

ξ = 1.0         # restitution coefficient
N = 2000        # number of particles
δt = 1/30       # time-step interval
T = 10          # time of the simulation
v0max = 10/20   # initial max-velocity
ncores = 8      # number of cores on PC

rm = 0.01
rm2 = 0.05
m = [10 1;]
r = [rm 1;]


# Some useful functions

# This function takes in the number of particles in the system, the masses of
# the particles, the radii of the particles and the maximum allowed initial
# velocity of the particles. Returns a Nx8 array for all the particles in the
# system.
function initialize_particles(N, masses, radii, v0max)
    Ψ = Array{Float64, 2}(undef, N, 8)
    θ = rand(Uniform(0, 2 * π), N)
    Ψ[:,3] = v0max * cos.(θ)        # velocity in x-direction
    Ψ[:,4] = v0max * sin.(θ)        # velocity in y-direction
    Ψ[:,7] = zeros(N)               # number of collisions
    Ψ[:,8] = zeros(N)               # time last collision occured

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

    @showprogress 1 "Initializing particles..." for i ∈ 1:1:N
        Ψ[i,1:2] = rand(Uniform(Ψ[i,5], 1 - Ψ[i,5]), 2)
    end
    return Ψ
end

function initialize_particles_improved(N, masses, radii, v0max)
    Ψ = Array{Float64, 2}(undef, N, 8)
    θ = rand(Uniform(0, 2 * π), N)
    Ψ[:,3] = v0max * cos.(θ)        # velocity in x-direction
    Ψ[:,4] = v0max * sin.(θ)        # velocity in y-direction
    Ψ[:,7] = zeros(N)               # number of collisions
    Ψ[:,8] = zeros(N)               # time last collision occured

    # Give particles masses
    n = 1
    for i in 1:1:(length(masses[:,1]))
        Ψ[n:(n + Int(N * masses[i,2]) - 1),6] .= masses[i,1]
        n += Int(N * masses[i,2])
    end
    # Give particles radii
    n = 1
    for i in 1:1:length(radii[:,1])
        Ψ[Int(n):Int(n + Int(N * radii[i,2]) - 1),5] .= radii[i,1]
        n += floor(N * radii[i,2])
    end

    @showprogress 1 "Initializing particles..." for i ∈ 1:1:N
        val = 1
        while val != 0
            Ψ[i,1:2] = rand(Uniform(Ψ[i,5], 1 - Ψ[i,5]), 2)
            val = 0
            for j ∈ 1:1:(i-1)
                val += check_overlap(Ψ[i,:], Ψ[j,:])
            end
        end
    end
    return Ψ
end

function initialize_two_colliding_particles(masses, radii, v0max, b)
    Ψ = Array{Float64, 2}(undef, 2, 8)
    Ψ[1,1:2] = [0.2 0.5]
    Ψ[2,1:2] = [0.8 0.5+b]
    Ψ[1,3:4] = [0 0]
    Ψ[2,3:4] = [-v0max 0]
    Ψ[:,7] = zeros(2)
    Ψ[:,8] = zeros(2)

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
    return Ψ
end

function initialize_projectile_and_wall(N, r, m, v0, kr, km)
    Ψ = Array{Float64, 2}(undef, N, 8)
    # Initialize projectile
    Ψ[1,1:2] = [0.5 0.75]
    Ψ[1,3:4] = [0 -v0]
    Ψ[1,5] = kr * r
    Ψ[1,6] = km * m
    Ψ[1,7:8] = [0 0]

    # Initialize the wall
    Ψ[2:end,3:4] .= [0 0]
    Ψ[2:end, 5] .= r
    Ψ[2:end, 6] .= m
    Ψ[2:end, 7:8] .= [0 0]
    @showprogress 1 "Initializing particles..." for i ∈ 2:1:N
        val = 1
        while val != 0
            Ψ[i,1] = rand(Uniform(Ψ[i,5], 1 - Ψ[i,5]))
            Ψ[i,2] = rand(Uniform(Ψ[i,5], 0.5 - Ψ[i,5]))
            val = 0
            for j ∈ 1:1:(i-1)
                val += check_overlap(Ψ[i,:], Ψ[j,:])
            end
        end
    end
    Ψ
end

# This function checks whether the two particles overlap. Returns 0 if the
# particles overlap and 1 if they dont
function check_overlap(Ψᵢ, Ψⱼ)
    Δx = vec(Ψⱼ[1:2] .- Ψᵢ[1:2])
    Rij = Ψⱼ[5] + Ψᵢ[5]

    if √(Δx ⋅ Δx) > Rij
        return 0
    else
        return 1
    end
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
function new_times_to_collision(Ψ, event, pq, c)
    N = length(Ψ[:,1])
    if event[6] != 3
        Δt_walls, nt = time_to_collision_with_walls(Ψ[Int(event[2]),:])
        if Δt_walls[1] != Inf
            pq[[event[1] + Δt_walls[1], Int(event[2]), 0, nt, 0, 1, c]] = event[1] + Δt_walls[1]
        end
        if Δt_walls[2] != Inf
            pq[[event[1] + Δt_walls[2], Int(event[2]), 0, nt, 0, 2, c]] = event[1] + Δt_walls[2]
        end
        for j in 1:1:N
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
            for j in 1:1:N
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

# Same as last function but using FLoops
function new_times_to_collision_parallel(Ψ, event, pq, c, ncores)
    N = length(Ψ[:,1])
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

# This function performes the event scheduled by the priority queue and returns
# the updated particle array, the updated PriorityQueue and updated number
# of collisions
function perform_event(Ψ, event, pq, c, ξ)
    if event[6] == 1 || event[6] == 2
        Ψ[Int(event[2]),:], happen = collision_with_wall(Ψ[Int(event[2]),:], ξ, event[6], event[4])
        if happen
            c += 1
            pq = new_times_to_collision(Ψ, event, pq, c)
            Ψ[Int(event[2]),8] = event[1]
        end
    else
        Ψᵢ = Ψ[Int(event[2]),:]
        Ψⱼ = Ψ[Int(event[3]),:]
        Ψᵢ, Ψⱼ, happen = collision_with_particle(Ψᵢ, Ψⱼ, ξ, event[4], event[5])
        Ψ[Int(event[2]),:] = Ψᵢ
        Ψ[Int(event[3]),:] = Ψⱼ
        if happen
            c += 1
            pq = new_times_to_collision(Ψ, event, pq, c)
            Ψ[Int(event[2]),8] = event[1]
            Ψ[Int(event[3]),8] = event[1]
        end
    end
    return Ψ, pq, c
end

function perform_event_parallel(Ψ, event, pq, c, ξ, ncores)
    if event[6] == 1 || event[6] == 2
        Ψ[Int(event[2]),:], happen = collision_with_wall(Ψ[Int(event[2]),:], ξ, event[6], event[4])
        if happen
            c += 1
            pq = new_times_to_collision_parallel(Ψ, event, pq, c, ncores)
            Ψ[Int(event[2]),8] = event[1]
        end
    else
        Ψᵢ = Ψ[Int(event[2]),:]
        Ψⱼ = Ψ[Int(event[3]),:]
        Ψᵢ, Ψⱼ, happen = collision_with_particle(Ψᵢ, Ψⱼ, ξ, event[4], event[5])
        Ψ[Int(event[2]),:] = Ψᵢ
        Ψ[Int(event[3]),:] = Ψⱼ
        if happen
            c += 1
            pq = new_times_to_collision_parallel(Ψ, event, pq, c, ncores)
            Ψ[Int(event[2]),8] = event[1]
            Ψ[Int(event[3]),8] = event[1]
        end
    end
    return Ψ, pq, c
end

function perform_event_parallel_with_tc(Ψ, event, pq, c, cpp, ξ, ncores, tc)
    if event[6] == 1 || event[6] == 2
        if event[1] - Ψ[Int(event[2]), 8] < tc
            ξ = 1.0
        end
        Ψ[Int(event[2]),:], happen = collision_with_wall(Ψ[Int(event[2]),:], ξ, event[6], event[4])
        if happen
            c += 1
            pq = new_times_to_collision_parallel(Ψ, event, pq, c, ncores)
            Ψ[Int(event[2]),8] = event[1]
        end
    else
        Ψᵢ = Ψ[Int(event[2]),:]
        Ψⱼ = Ψ[Int(event[3]),:]
        if event[1] - Ψᵢ[8] < tc || event[1] - Ψⱼ[8] < tc
            ξ = 1.0
        end
        Ψᵢ, Ψⱼ, happen = collision_with_particle(Ψᵢ, Ψⱼ, ξ, event[4], event[5])
        Ψ[Int(event[2]),:] = Ψᵢ
        Ψ[Int(event[3]),:] = Ψⱼ
        if happen
            c += 1
            cpp += 1
            pq = new_times_to_collision_parallel(Ψ, event, pq, c, ncores)
            Ψ[Int(event[2]),8] = event[1]
            Ψ[Int(event[3]),8] = event[1]
        end
    end
    return Ψ, pq, c, cpp
end

# This function animates a set of particle tracks modelling a homogeneous gas.
function animate_homogeneous_gas(particle_tracks, rm)
    particle_x = particle_tracks[:,1:2:end]
    particle_y = particle_tracks[:,2:2:end]

    n = Int(length(particle_tracks[1,:]) / 2)

    gr()
    anim = @animate for i ∈ 1:n
        plot(particle_x[1:end,i], particle_y[1:end,i], seriestype= :scatter,
         xlims=(0,1), ylims=(0,1), legend = false,
         markersize=rm*550*1.101, size=(600, 600))
    end

    gif(anim, "Scatterplotp.gif", fps=30)
end

# This function counts the amount of particles that have been affected
function affected_amount(Ψ₀, Ψ₁, tol)
    ap = 0
    for i ∈ 2:Int(length(Ψ₀[:,1]))
        Δx = vec(Ψ₀[i,1:2] .- Ψ₁[i,1:2])
        if sqrt(Δx ⋅ Δx) > tol
            ap += 1
        end
    end
    return ap
end

# These functions simulate with a recording of the particles position for a
# constant timestep. These functions are ideal for animation of the system.
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
                    pq = new_times_to_collision(Ψ, event, pq)
                    Ψ[Int(event[2]),8] = event[1]
                end
            else
                Ψᵢ = Ψ[Int(event[2]),:]
                Ψⱼ = Ψ[Int(event[3]),:]
                Ψᵢ, Ψⱼ, happen = collision_with_particle(Ψᵢ, Ψⱼ, ξ, event[4], event[5])
                Ψ[Int(event[2]),:] = Ψᵢ
                Ψ[Int(event[3]),:] = Ψⱼ
                if happen
                    coll += 1
                    pq = new_times_to_collision(Ψ, event, pq)
                    Ψ[Int(event[2]),8] = event[1]
                    Ψ[Int(event[3]),8] = event[1]
                end
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
    Ψ = initialize_particles_improved(N, masses, radii, v0max)
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

            Ψ, pq, coll = perform_event_parallel(Ψ, event, pq, coll, ξ, ncores)

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


# This function is to solve test B4
function test_B4(N, ncores)
    r₁ = 0.1
    m₁ = 1e6
    r₂ = 0.001
    m₂ = 1
    masses = [m₁ 0.5; m₂ 0.5]
    radii = [r₁ 0.5; r₂ 0.5]

    bs = LinRange(0, r₁ + r₂, N)
    θ = Array{Float64, 1}(undef, N)

    for i in 1:1:N
        Ψ = initialize_two_colliding_particles(masses, radii, 5, bs[i])
        vᵢ = Array{Float64, 2}(undef, 2, 2)
        vᵢ[1,:] = Ψ[2,3:4]

        pq = PriorityQueue{Array{Float64, 1}, Float64}()
        pq = initial_collisiontimes_parallel(Ψ, pq, 2, ncores)

        event = dequeue!(pq)
        coll_count = 0
        Ψ[:,1:2] = Ψ[:,1:2] .+ Ψ[:,3:4] * event[1]

        Ψ, pq, coll_count = perform_event_parallel(Ψ, event, pq, coll_count, 1, ncores)

        vᵢ[2,:] = Ψ[2,3:4]

        θ[i] = acos((vᵢ[1,:] ⋅ vᵢ[2,:])/(sqrt(vᵢ[1,:] ⋅ vᵢ[1,:]) * sqrt(vᵢ[2,:] ⋅ vᵢ[2,:])))
    end
    return (bs ./ (r₁ + r₂)), θ
end

# These functions are to solve Problem 1
function problem_1(N, ξ, masses, radii, v0max, it)
    Ψ = initialize_particles(N, masses, radii, v0max)

    t = 0

    velocities_i = sqrt.(Ψ[:,3].^2 .+ Ψ[:,4].^2)
    velocities_f = Array{Float64, 1}(undef, N)

    pq = PriorityQueue{Array{Float64, 1}, Float64}()
    pq = initial_collisiontimes(Ψ, pq, N)

    event = dequeue!(pq)
    coll_count = 0

    @showprogress 1 "Computing..." for i in 2:(it + 1)
        Δt = event[1] - t
        Ψ[:,1:2] = Ψ[:,1:2] .+ Ψ[:,3:4] * Δt

        Ψ, pq, coll_count = perform_event(Ψ, event, pq, coll_count, ξ)

        t = event[1]
        event = dequeue!(pq)
    end
    display("Returned")

    velocities_f = sqrt.(Ψ[:,3].^2 .+ Ψ[:,4].^2)
    return velocities_i, velocities_f
end

function problem_1p(N, ξ, masses, radii, v0max, it, ncores)
    Ψ = initialize_particles(N, masses, radii, v0max)

    t = 0

    velocities_i = sqrt.(Ψ[:,3].^2 .+ Ψ[:,4].^2)
    velocities_f = Array{Float64, 1}(undef, N)

    pq = PriorityQueue{Array{Float64, 1}, Float64}()
    pq = initial_collisiontimes_parallel(Ψ, pq, N, ncores)

    event = dequeue!(pq)
    coll_count = 0

    @showprogress 1 "Computing..." for i in 2:(it + 1)
        Δt = event[1] - t
        Ψ[:,1:2] = Ψ[:,1:2] .+ Ψ[:,3:4] * Δt

        Ψ, pq, coll_count = perform_event_parallel(Ψ, event, pq, coll_count, ξ, ncores)

        t = event[1]
        event = dequeue!(pq)
    end
    display("Returned")

    velocities_f = sqrt.(Ψ[:,3].^2 .+ Ψ[:,4].^2)
    return velocities_i, velocities_f
end


function generate_statistics(it, n, ncores)
    for i ∈ 1:1:n
        vel = problem_1p(N, ξ, m, r, v0max, it, ncores)
        filename = "velocities_parallel" * string(i) * ".csv"
        writedlm(filename, vel, ',')
        display(filename)
    end
end


# This is the functions to solve Problem 2 in the exercise

function problem_2p(N, ξ, m, r, v0max, it, ncores)
    masses = [m 0.5; 4*m 0.5]
    radii = [r 1;]
    Ψ = initialize_particles_improved(N, masses, radii, v0max)

    t = 0
    times = Array{Float64, 1}(undef, it)
    times[1] = 0

    velocities_1 = Array{Float64, 1}(undef, N)
    velocities_2 = Array{Float64, 1}(undef, N)
    velocities_1[1:Int(trunc(N/2))] = sqrt.(Ψ[1:Int(trunc(N/2)),3].^2 .+ Ψ[1:Int(trunc(N/2)),4].^2)
    velocities_2[1:Int(trunc(N/2))] = sqrt.(Ψ[Int(trunc(N/2))+1:end,3].^2 .+ Ψ[Int(trunc(N/2))+1:end,4].^2)

    avg_speed_1 = Array{Float64, 1}(undef, it)
    avg_speed_2 = Array{Float64, 1}(undef, it)
    avg_speed_1[1] = mean(sqrt.(Ψ[1:Int(trunc(N/2)),3].^2 .+ Ψ[1:Int(trunc(N/2)),4].^2))
    avg_speed_2[1] = mean(sqrt.(Ψ[Int(trunc(N/2))+1:end,3].^2 .+ Ψ[Int(trunc(N/2))+1:end,4].^2))

    avg_K_1 = Array{Float64, 1}(undef, it)
    avg_K_2 = Array{Float64, 1}(undef, it)
    avg_K_1[1] = (1/2) * masses[1,1] * avg_speed_1[1]^2
    avg_K_2[1] = (1/2) * masses[2,1] * avg_speed_2[1]^2

    pq = PriorityQueue{Array{Float64, 1}, Float64}()
    pq = initial_collisiontimes_parallel(Ψ, pq, N, ncores)

    event = dequeue!(pq)
    coll = 0

    @showprogress 1 "Computing..." for i ∈ 2:it
        Δt = event[1] - t
        Ψ[:,1:2] = Ψ[:,1:2] .+ Ψ[:,3:4] * Δt
        Ψ, pq, coll = perform_event_parallel(Ψ, event, pq, coll, ξ, ncores)

        avg_speed_1[i] = mean(sqrt.(Ψ[1:Int(trunc(N/2)),3].^2 .+ Ψ[1:Int(trunc(N/2)),4].^2))
        avg_speed_2[i] = mean(sqrt.(Ψ[Int(trunc(N/2))+1:end,3].^2 .+ Ψ[Int(trunc(N/2))+1:end,4].^2))
        avg_K_1[i] = (1/2) * masses[1,1] * avg_speed_1[i]^2
        avg_K_2[i] = (1/2) * masses[2,1] * avg_speed_2[i]^2

        times[i] = event[1]
        t = event[1]
        event = dequeue!(pq)
    end
    display("Returned")
    velocities_1[Int(trunc(N/2))+1:end] = sqrt.(Ψ[1:Int(trunc(N/2)),3].^2 .+ Ψ[1:Int(trunc(N/2)),4].^2)
    velocities_2[Int(trunc(N/2))+1:end] = sqrt.(Ψ[Int(trunc(N/2))+1:end,3].^2 .+ Ψ[Int(trunc(N/2))+1:end,4].^2)

    return velocities_1, velocities_2, avg_speed_1, avg_speed_2, avg_K_1, avg_K_2, times
end


# This is the function to solve problem 3 in the exercise. Run code at 300000
# iterations with ξ at 1, 0.9 and 0.8. No need for TC actually.

function problem_3p(N, ξ, m, r, v0max, it, ncores, tc, eng_maxcpp = false, maxcpp = 0)
    masses = [m 0.5; 4*m 0.5]
    radii = [r 1;]
    Ψ = initialize_particles_improved(N, masses, radii, v0max)

    t = 0
    times = Array{Float64, 1}(undef, it)
    times[1] = 0

    avg_K_1 = Array{Float64, 1}(undef, it)
    avg_K_2 = Array{Float64, 1}(undef, it)
    avg_K_tot = Array{Float64, 1}(undef, it)
    avg_K_1[1] = (1/2) * masses[1,1] * mean(Ψ[1:Int(trunc(N/2)),3].^2 .+ Ψ[1:Int(trunc(N/2)),4].^2)
    avg_K_2[1] = (1/2) * masses[2,1] * mean(Ψ[Int(trunc(N/2))+1:end,3].^2 .+ Ψ[Int(trunc(N/2))+1:end,4].^2)
    avg_K_tot[1] = (avg_K_1[1] + avg_K_2[1]) / 2

    pq = PriorityQueue{Array{Float64, 1}, Float64}()
    pq = initial_collisiontimes_parallel(Ψ, pq, N, ncores)

    event = dequeue!(pq)
    coll = 0
    collpp = 0
    it_stop = 0

    @showprogress 1 "Computing..." for i ∈ 2:it
        Δt = event[1] - t
        Ψ[:,1:2] = Ψ[:,1:2] .+ Ψ[:,3:4] * Δt
        Ψ, pq, coll, collpp = perform_event_parallel_with_tc(Ψ, event, pq, coll, collpp, ξ, ncores, tc)

        avg_K_1[i] = (1/2) * masses[1,1] * mean(Ψ[1:Int(trunc(N/2)),3].^2 .+ Ψ[1:Int(trunc(N/2)),4].^2)
        avg_K_2[i] = (1/2) * masses[2,1] * mean(Ψ[Int(trunc(N/2))+1:end,3].^2 .+ Ψ[Int(trunc(N/2))+1:end,4].^2)
        avg_K_tot[i] = (avg_K_1[i] + avg_K_2[i]) / 2

        times[i] = event[1]
        t = event[1]
        event = dequeue!(pq)

        if collpp / N > maxcpp && eng_maxcpp
            it_stop = i
            break
        end
    end
    display("Returned")
    return avg_K_1, avg_K_2, avg_K_tot, times, it_stop
end


# This function solves problem 4.

function problem_4p(N, ξ, r, m, v0, kr, km, tol, tc, it, ncores)
    Ψ = initialize_projectile_and_wall(N, r, m, v0, kr, km)
    E = Array{Float64, 1}(undef, it)
    E[1] = (1/2) * km * m * v0^2
    Ψ₀ = copy(Ψ)
    pac = (N - 1) * π * r^2 / (0.5 * 1)

    t = 0

    pq = PriorityQueue{Array{Float64, 1}, Float64}()
    pq = initial_collisiontimes_parallel(Ψ, pq, N, ncores)

    event = dequeue!(pq)
    coll = 0
    collpp = 0
    it_stop = 0

    @showprogress 1 "Computing..." for i ∈ 2:it
        Δt = event[1] - t
        Ψ[:,1:2] = Ψ[:,1:2] .+ Ψ[:,3:4] * Δt

        Ψ, pq, coll, collpp = perform_event_parallel_with_tc(Ψ, event, pq, coll, collpp, ξ, ncores, tc)
        E[i] = (1/2) * km * m * (Ψ[1,3]^2 + Ψ[1,4]^2) + (1/2) * m * sum(Ψ[2:end,3].^2 .+ Ψ[2:end,4].^2)

        t = event[1]
        event = dequeue!(pq)

        if E[i] < 0.1 * E[1]
            it_stop = i
            break
        end
    end

    display("Returned")
    # Returnes the fraction of affected particles
    return Ψ₀[:,1:2], Ψ[:,1:2], affected_amount(Ψ₀, Ψ, tol), pac
end

function scan_variable_v(tol, ncores)
    tol = 0
    r = 0.005
    N = 3200
    v = [5 * i^2 for i ∈ 5:1:15]
    aps = Array{Float64, 1}(undef, Int(length(v)))
    for i in 1:1:Int(length(v))
        init_pos, final_pos, ap, pac = problem_4p(N, 0.5, r, 10, v[i], 5, 25, tol, 0, 100000, ncores)
        aps[i] = ap
    end
    writedlm("craters_v.csv", aps, ',')
    writedlm("craters_vv.csv", v, ',')
end

# For test B4
#bs, θ = test_B4(100, ncores)
#plot(bs, θ, yticks=([0 π/4 π/2 3*π/4 π], [L"0" L"\frac{\pi}{4}" L"\frac{\pi}{2}" L"\frac{3 \pi}{4}" L"\pi"]))
#savefig("testB4.pdf")

# For problem 1:
#for i ∈ 1:1:20
#    veli, velf = problem_1p(N, ξ, m, r, v0max, 1000000, ncores)
#    fn = "final_velocities_parallel"*string(i)*".csv"
#    writedlm(fn, velf, ',')
#    fn = "initial_velocities_parallel"*string(i)*".csv"
#    writedlm(fn, veli, ',')
#end



# For problem 3:
#K1, K2, Kt, ti, it_stop = problem_3p(N, 0.8, 10, 0.001, v0max, 300000, ncores, 0)
#plot([ti ti ti], [K1, K2, Kt])

# For problem 4:
#tol = 0
#r = 0.005
#N = 3200
#init_pos, final_pos, ap, pac = problem_4p(N, 0.5, r, 10, 5, 5, 25, tol, 0, 100000, ncores)

#display(ap)
#display(pac)
#plot(final_pos[1:1,1], final_pos[1:1,2], seriestype = :scatter, legend = false, size = (600, 600),
#markersize = 5 * r * (1.101 * 550),  xlims=(0,1), ylims=(0,1))
#plot!(final_pos[2:end,1], final_pos[2:end,2], seriestype = :scatter,
#markersize = r * (1.101 * 550))

#scan_variable_v(tol, ncores)

#velsq = vel.^2
#E = transpose(sum(velsq, dims=1))
#avg_vel = transpose(mean(vel, dims=1))

#writedlm("times.csv", ti, ',')

#particle_track, velocities = mainp(N, ξ, m, r, v0max, T, δt, ncores)
#particle_track, velocities = mainp(N, ξ, m, r, v0max, T, δt, ncores)
#animate_homogeneous_gas(particle_tracks, rm)

#v1, v2, av1, av2, k1, k2, t = problem_2p(N, ξ, 10, 0.001, v0max, 1000000, ncores)

#writedlm("vels3.csv", vcat(v1, v2), ',')
#writedlm("avg_vels3.csv", vcat(av1, av2), ',')
#writedlm("avg_kin3.csv", vcat(k1, k2), ',')
#writedlm("time3.csv", t, ',')
