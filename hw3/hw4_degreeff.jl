using MPI
using Plots
using Statistics

# Setup

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nproc = MPI.Comm_size(comm)
rank_s = mod(rank + 1, nproc)
rank_r = mod(rank - 1, nproc)

MPI.Barrier(comm)
rank == 0 && println("FINISHED INITIALIZATION")

# Data Collection for Serial Bandwidth

const N = 25
const N_SAMPLES = 1000
const sizes = [2^n for n ∈ 1:N]

const send = (s -> rand(Int8, s)).(sizes)
recv = similar.(send)
times = Array{Float64}(undef, N)
samples = Array{Float64}(undef, N_SAMPLES)

for i ∈ 1:N
    for j ∈ 1:N_SAMPLES
        samples[j] = @elapsed begin
            sreq = MPI.Isend(send[i], rank_r, (i - 1) * nproc + rank, comm)
            rreq = MPI.Irecv!(recv[i], rank_s, (i - 1) * nproc + rank_s, comm)
            MPI.Waitall!([sreq, rreq])
        end
    end
    times[i] = median(samples)
end

MPI.Barrier(comm)
rank == 0 && println("FINISHED GENERATING DATA")

tag_base = N * nproc
if rank == 0
    all_times = [similar(times) for _ ∈ 1:nproc]
    all_times[1] = times
    for i in 2:nproc
        all_times[i], statrcv = MPI.recv(i - 1, tag_base + (i - 1), comm)
    end
else
    sreq = MPI.send(times, 0, tag_base + rank, comm)
end


MPI.Barrier(comm)
rank == 0 && println("FINISHED GATHERING DATA")

# Data Analysis

if rank == 0
    latency = plot(ylabel="Latency (s)", xlabel="Message Size (bytes)", yaxis=:log, xaxis=:log, legend=:bottomright)
    bandwidth = plot(ylabel="Bandwidth (bytes/s)", xlabel="Message Size (bytes)", yaxis=:log, xaxis=:log, legend=:bottomright)
    for i ∈ 1:nproc
        scatter!(latency, sizes, all_times[i], label="rank $(i - 1)")
        scatter!(bandwidth, sizes, sizes ./ all_times[i], label="rank $(i - 1)")
    end
    savefig(latency, "latency.png")
    savefig(bandwidth, "bandwidth.png")
end

MPI.Barrier(comm)
rank == 0 && println("FINISHED ANALYZING DATA")
