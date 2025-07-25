# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NCCL (pronounced "Nickel") is NVIDIA's library of optimized primitives for inter-GPU communication. It implements collective operations like all-reduce, all-gather, reduce, broadcast, and reduce-scatter, optimized for multi-GPU systems using PCIe, NVLink, NVswitch, and various network interconnects.

## Build System

### Primary Build Commands
```bash
# Build the library (most common)
make -j src.build

# Build with custom CUDA path
make src.build CUDA_HOME=/path/to/cuda

# Build for specific GPU architecture (faster compilation)
make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

# Install system packages
make pkg.debian.build  # For Debian/Ubuntu
make pkg.redhat.build  # For RedHat/CentOS
make pkg.txz.build     # OS-agnostic tarball
```

### Build Configuration
- **BUILDDIR**: Build output directory (default: `./build/`)
- **DEBUG**: Set to 1 for debug builds with `-O0 -G -g`
- **VERBOSE**: Set to 1 for verbose compilation output
- **CUDA_HOME**: CUDA installation path (default: `/usr/local/cuda`)
- **NVCC_GENCODE**: GPU architectures to compile for (auto-detected from CUDA version)

### Key Build Artifacts
- `build/lib/libnccl.so.X.Y.Z` - Main shared library
- `build/lib/libnccl_static.a` - Static library
- `build/include/nccl.h` - Public API header
- `build/bin/ncclras` - RAS (Reliability, Availability, Serviceability) client

## Architecture Overview

### Core Components
- **src/collectives.cc** - Main collective operation implementations
- **src/transport/** - Communication transport layer (IB, TCP, shared memory, P2P)
- **src/device/** - GPU kernel implementations for collective operations
- **src/graph/** - Topology detection and optimization (rings, trees, tuning)
- **src/plugin/** - Plugin system for network, profiler, and tuner extensions

### Collective Operations Implementation

**Operation Flow:**
1. **API Layer** (`src/collectives.cc`): Public APIs like `ncclAllReduce()` create `ncclInfo` structs containing operation metadata
2. **Enqueue Layer** (`src/enqueue.cc`): Validates parameters, manages kernel launches, handles CUDA stream synchronization
3. **Planning Layer**: Determines optimal algorithms, protocols, and data partitioning
4. **Execution Layer**: Launches generated device kernels with transport-specific communication

**Key Constants:**
- `ALLREDUCE_CHUNKSTEPS = NCCL_STEPS/2` - Pipeline depth for AllReduce operations
- `ALLREDUCE_SLICESTEPS = NCCL_STEPS/4` - Data slicing granularity
- `NCCL_MAX_OPS = 2048` - Maximum operations in flight
- `NCCL_STEPS = 8` - Default pipeline depth

### Transport Layer Architecture

**Transport Selection Logic** (`src/transport.cc`):
```
Transport Priority: P2P > SHM > NET > COLLNET > PROFILER
```

**Transport Types:**
1. **P2P Transport** (`transport/p2p.cc`):
   - `P2P_DIRECT`: Direct GPU memory access via CUDA IPC
   - `P2P_INTERMEDIATE`: Copy engine assisted transfers
   - `P2P_IPC`: Inter-process GPU memory sharing
   - `P2P_CUMEM`: CUDA memory handle based transfers

2. **Network Transport** (`transport/net_ib.cc`, `transport/net_socket.cc`):
   - **InfiniBand**: RDMA-based with memory registration caching
   - **TCP Sockets**: Fallback network transport
   - **Memory Registration Cache**: `ncclIbMrCache` for RDMA efficiency

3. **Shared Memory** (`transport/shm.cc`):
   - Intra-node communication via shared memory segments
   - Lock-free ring buffers for producer-consumer patterns

### Device Code Generation System

**Code Generation Pipeline** (`src/device/generate.py`):
1. **Template Instantiation**: Creates specialized kernels for each (collective, reduction_op, datatype, algorithm, protocol) tuple
2. **Equivalence Classes**: Maps similar functions to shared implementations (e.g., signed→unsigned int for sum/prod)
3. **Kernel Specialization**: Generates optimized kernels using `best_kernel()` heuristics
4. **CUDA Architecture Filtering**: Only generates kernels supported by target GPU architectures

**Protocol Implementations:**
- **ProtoSimple**: Direct memory transfers, configurable unroll factors
- **ProtoLL**: Low-latency with inline flags, 8-byte data granularity  
- **ProtoLL128**: 128-bit optimized with 16-byte operations

**Kernel Templates** (`src/device/common_kernel.h`):
- `reduceCopyPacks()`: Core loop for data movement and reduction
- Vectorized operations with `BytePack<N>` for memory efficiency
- Warp-level coordination with configurable unroll factors

### AllReduce Implementation Deep Dive

**API Entry Point** (`src/collectives.cc:93-102`):
```c
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream,
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
```

**Algorithm Selection** (`src/enqueue.cc:1972-1980`):
- **RING**: Default algorithm, bandwidth-optimal with `2*(N-1)` steps
- **TREE**: Latency-optimal for small messages with `2*log(N)` steps  
- **COLLNET_DIRECT**: In-network reduction using SHARP hardware
- **NVLS**: NVLink switch acceleration for single/multi-node
- **NVLS_TREE**: Hybrid approach combining NVLS + tree algorithms

**RING Algorithm** (`src/device/all_reduce.h:13-83`):
1. **Reduce-Scatter Phase**: Each rank reduces its assigned chunk while forwarding data in ring
2. **All-Gather Phase**: Broadcast final reduced chunks to all ranks
3. **Data Flow**: Each element traverses ring exactly twice (optimal bandwidth)
4. **Primitives**: `directRecvReduceDirectSend()`, `directRecvCopyDirectSend()`

**TREE Algorithm** (`src/device/all_reduce.h:86-146`):
1. **Reduce-Up Phase**: Leaf nodes send to parents, intermediate nodes reduce and forward
2. **Broadcast-Down Phase**: Root broadcasts final result down the tree
3. **Topology**: Binary tree with `NCCL_MAX_TREE_ARITY = 3` children maximum
4. **Optimization**: Uses `FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>` for efficient fan-in/out

**NVLS Algorithm** (`src/device/all_reduce.h:388-484`):
1. **Thread Allocation**: Specialized warps for scatter, gather, reduce, broadcast phases
2. **Single-Node**: Direct NVLink switch reduction with multicast memory
3. **Multi-Node**: Combines local NVLS reduction with inter-node communication
4. **Hardware Requirements**: SM90+ GPUs, CUDA 12.1+, NVLink switch multicast support

**COLLNET_DIRECT Algorithm** (`src/device/all_reduce.h:252-384`):
1. **Scatter Phase**: Distribute data to network reduction heads
2. **Network Reduction**: InfiniBand SHARP performs reduction in-network
3. **Gather Phase**: Collect reduced results from network
4. **Limitations**: Maximum `NCCL_MAX_DIRECT_ARITY = 7` participants per operation

### Algorithm Implementations

**COLLNET_DIRECT** (`src/transport/coll_net.cc`):
- **Purpose**: In-network collective operations using specialized network hardware
- **Hardware**: InfiniBand switches with SHARP technology or similar offload engines
- **Topology**: Direct communication pattern, limited to `NCCL_MAX_DIRECT_ARITY = 7` participants
- **Operations**: Network-offloaded AllReduce, AllGather, ReduceScatter
- **Benefits**: Reduces GPU computation load, improved bandwidth utilization for multi-node scenarios

**NVLS (NVLink SHARP)** (`src/transport/nvls.cc`):
- **Purpose**: Hardware-accelerated collectives using NVLink switches
- **Requirements**: CUDA 12.1+, SM90+ GPUs, NVLink switch hardware with multicast support
- **Memory**: CUDA multicast groups with unified memory (`cuMulticastCreate()`)
- **Channels**: Architecture-specific (16 for Hopper, 24-32 for Blackwell)
- **Operations**: Optimized for AllReduce (Sum, MinMax), AllGather, ReduceScatter
- **Topology**: Tree-based with NVLink switch acceleration, supports larger GPU counts

### Plugin Architecture
NCCL supports three types of plugins:
- **Network plugins** (`ext-net/`) - Custom network transport implementations with versioned API (`ncclNet_v10`)
- **Profiler plugins** (`ext-profiler/`) - Performance profiling and monitoring hooks
- **Tuner plugins** (`ext-tuner/`) - Algorithm and parameter tuning based on measured performance

## Code Organization

### Header Files
- **src/include/core.h** - Core NCCL definitions and API macros
- **src/nccl.h.in** - Template for public API header (processed during build)
- **src/include/device.h** - Device-side API definitions
- **src/include/collectives.h** - Collective operation interfaces

### Transport Layer
- **net_ib.cc** - InfiniBand RDMA transport
- **net_socket.cc** - TCP/IP socket transport  
- **shm.cc** - Shared memory transport
- **p2p.cc** - GPU peer-to-peer transport
- **nvls.cc** - NVIDIA Link Switch transport

### Memory Management
- **src/allocator.cc** - Custom memory allocator
- **src/register/** - Memory registration for RDMA operations

## Topology Detection and Algorithm Selection

### Topology Graph Construction (`src/graph/topo.cc`)
**Node Types:**
- `GPU`: Graphics processing units
- `PCI`: PCI switches and bridges  
- `NVS`: NVLink switches
- `CPU`: CPU nodes
- `NIC`: Network interface cards
- `NET`: Network fabric

**Link Types:**
- `LOC`: Local/same device
- `NVL`: NVLink
- `C2C`: Chip-to-chip
- `PCI`: PCIe links
- `SYS`: Inter-CPU/system links
- `NET`: Network links

**Bandwidth Detection:**
- CPU architecture-specific bandwidth estimation (Power9, ARM, Intel, AMD)
- PCI topology analysis for GPU interconnect discovery
- NVLink detection and speed classification

### Algorithm Selection Logic (`src/graph/tuning.cc`)
**Parameter Parsing:**
- Supports prefix-based configuration: `NCCL_ALGO="ring,collnetdirect;allreduce:tree"`
- Negation syntax: `NCCL_PROTO="^LL128;allreduce:LL128"`
- Per-operation overrides with semicolon separation

**Thread Count Optimization:**
- `NCCL_NTHREADS`: General thread count (must be multiple of `WARP_SIZE=32`)
- `NCCL_LL128_NTHREADS`: Protocol-specific thread optimization
- Auto-tuning based on GPU architecture and memory hierarchy

## Development Workflow

### Advanced Build Options
```bash
# Filter kernel generation for faster development builds
make ONLY_FUNCS="AllReduce Sum f32 * *"

# Build with specific features
make DEBUG=1 VERBOSE=1 TRACE=1        # Debug build with tracing
make NVTX=0                           # Disable NVTX profiling markers
make ASAN=1                           # Address sanitizer build
make GCOV=1 DEBUG=1                   # Code coverage build
```

### Testing
Tests are maintained in a separate repository:
```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

### Debugging and Profiling
**Logging Configuration:**
- `NCCL_DEBUG=INFO` - Detailed operation logging
- `NCCL_DEBUG_SUBSYS=COLL,P2P,NET` - Subsystem-specific debug output
- `NCCL_DEBUG_FILE=debug.log` - Log to file instead of stderr

**Performance Analysis:**
- `NCCL_ALGO` - Force algorithm: `TREE`, `RING`, `COLLNET_DIRECT`, `COLLNET_CHAIN`, `NVLS`, `NVLS_TREE`
- `NCCL_PROTO` - Force protocol: `LL`, `LL128`, `SIMPLE`  
- `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` - Control channel parallelism
- `NCCL_TREE_THRESHOLD` - Algorithm switch point (bytes)
- `NCCL_LL_THRESHOLD` / `NCCL_LL128_THRESHOLD` - Protocol switch points

**Algorithm-Specific Tuning:**
- `NCCL_NVLS_ENABLE` - Control NVLS algorithm (0=disable, 1=force, 2=auto-detect)
- `NCCL_NVLS_CHUNKSIZE` - NVLS chunk size (default: 128KB)
- `NCCL_COLLNET_ENABLE` - Enable/disable CollNet algorithms
- `NCCL_NET` - Force specific network implementation for CollNet

**Memory and Transport Debugging:**
- `NCCL_SHM_DISABLE=1` - Disable shared memory transport
- `NCCL_P2P_DISABLE=1` - Disable P2P transport
- `NCCL_NET_DISABLE=1` - Disable network transport
- `NCCL_IB_DISABLE=1` - Disable InfiniBand specifically

## Version Management

Current version is defined in `makefiles/version.mk`:
- NCCL_MAJOR: 2
- NCCL_MINOR: 27  
- NCCL_PATCH: 5

Version information is embedded in build artifacts through template processing of `src/nccl.h.in`.

## Important Development Notes

### Performance Characteristics
- **Memory Bandwidth**: NCCL optimizes for memory-bound workloads using vectorized operations (`BytePack<16>`)
- **Pipeline Depth**: `NCCL_STEPS=8` provides optimal latency/bandwidth tradeoff for most workloads
- **Warp Coordination**: Device kernels coordinate at warp boundaries (`WARP_SIZE=32`) for efficiency
- **Protocol Selection**: LL for latency-critical small messages, LL128 for medium sizes, SIMPLE for large transfers
- **Algorithm Selection**: Ring for bandwidth-optimal large transfers, Tree for latency-optimal small messages, NVLS/CollNet for hardware acceleration

### AllReduce Performance Tuning
- **Ring vs Tree Threshold**: Controlled by `NCCL_TREE_THRESHOLD` (default switches based on message size)
- **NVLS Optimization**: Uses specialized thread allocation with separate warps for scatter/gather/reduce phases
- **CollNet Chunking**: Dynamic chunk size optimization based on network depth and bandwidth
- **Memory Access**: All algorithms use 128-bit vectorized operations for maximum memory bandwidth

### Code Generation Details  
- **Kernel Filtering**: Use `ONLY_FUNCS` to reduce compilation time during development
- **Equivalence Classes**: Similar operations share kernels (signed/unsigned integers for sum/prod)
- **Architecture Support**: Kernels automatically filtered based on CUDA architecture capabilities
- **Template Specialization**: `best_kernel()` function controls which kernel variants are specialized
- **Algorithm Coverage**: Each collective supports specific algorithms (e.g., AllReduce: TREE, RING, COLLNET_DIRECT, COLLNET_CHAIN, NVLS, NVLS_TREE)

### Data Movement Primitives (`src/device/primitives.h`)
- **`directSend()`**: Direct peer-to-peer data transfer between GPUs
- **`directRecvReduceDirectSend()`**: Atomic receive, reduce, and forward operation
- **`directRecvReduceCopyDirectSend()`**: Receive, reduce, copy to output buffer, and forward
- **`scatter()`/`gather()`**: Multi-destination data distribution and collection
- **`FanAsymmetric<MaxRecv, MaxSend>`**: Template for asymmetric communication patterns
- **`ProtoSimple`/`ProtoLL`/`ProtoLL128`**: Protocol-specific optimizations for different message sizes

### Transport Layer Details
- **Transport Selection**: Automatic fallback from P2P→SHM→NET based on topology and capability
- **Memory Registration**: InfiniBand transport uses `ncclIbMrCache` for efficient memory registration
- **Connection Management**: Transport setup follows listen→connect→accept pattern for reliability  
- **Buffer Management**: Ring buffers with producer-consumer semantics for lock-free operation

### Plugin System Architecture
- **Network Plugin API**: Versioned interfaces (`ncclNet_v10`) allow backward compatibility
- **Plugin Discovery**: Uses `libnccl-net.so` or `libnccl-net-${NCCL_NET_PLUGIN}.so` naming
- **Device Offload**: Plugins can support GPU-side networking for zero-copy operations
- **Multi-Rail**: Virtual device support allows aggregating multiple physical NICs
- **CollNet Integration**: Network plugins can provide `collNet` API for in-network collective operations

### Hardware Acceleration Features
- **NVLS Requirements**: CUDA 12.1+, SM90+ GPUs, NVLink switch multicast support (`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`)
- **CollNet Requirements**: Network hardware with collective offload (e.g., InfiniBand SHARP)
- **Memory Alignment**: NVLS uses 2MB alignment (`NVLS_MEM_ALIGN_SIZE`) for optimal performance
- **Channel Scaling**: NVLS channels scale with architecture (16→32 channels from Hopper→Blackwell)

### Debugging and Instrumentation
- **NVTX Integration**: Built-in profiling markers for GPU timeline analysis (disable with `NVTX=0`)
- **Memory Debugging**: AddressSanitizer support with proper CUDA shared memory handling
- **Proxy Operations**: Background CPU threads handle network operations while GPU computes
- **RAS System**: Reliability monitoring via `ncclras` client for production deployments

## Common File Patterns and Naming

- `.cc` files - C++ source code (host-side implementation)
- `.cu` files - CUDA source code (device-side kernels)  
- `.h` files in `src/include/` - Internal headers with device/host shared definitions
- `Makefile` hierarchy - Recursive build system with common configuration in `makefiles/`
- `*.in` files - Templates processed during build (version substitution, configuration)
- `generate.py` - Python scripts for automated code generation
- `*wrap.h` - Dynamic library loading wrappers (e.g., `ibvwrap.h`, `cudawrap.h`)