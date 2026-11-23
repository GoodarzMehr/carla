#! /bin/bash

# Builds libcarla_voxel_gpu, the CUDA path of the voxel detection sensor
# (use_gpu), into the plugin's CarlaDependencies, where the package picks it
# up. Without nvcc it keeps a library that is up to date and removes one
# that is not, so a package never carries code older than its source; the
# sensor then reports use_gpu as unavailable.

set -e

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SOURCE=${ROOT}/Unreal/CarlaUE4/Plugins/Carla/Source/VoxelGpu/VoxelGpu.cu
HEADER=${ROOT}/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Sensor/VoxelGpuApi.h
OUTPUT=${ROOT}/Unreal/CarlaUE4/Plugins/Carla/CarlaDependencies/lib/libcarla_voxel_gpu.so
BUILD=${ROOT}/Build/voxel-gpu-build

if [ -f "${OUTPUT}" ] && [ "${OUTPUT}" -nt "${SOURCE}" ] && [ "${OUTPUT}" -nt "${HEADER}" ] && [ "${OUTPUT}" -nt "$0" ]; then
  echo "BuildVoxelGpu.sh: ${OUTPUT} is up to date."
  exit 0
fi

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
NVCC=${CUDA_HOME}/bin/nvcc

if [ ! -x "${NVCC}" ]; then
  NVCC=$(command -v nvcc || true)
fi

if [ -z "${NVCC}" ]; then
  if [ -f "${OUTPUT}" ]; then
    echo "BuildVoxelGpu.sh: nvcc not found; removing ${OUTPUT}, which is older than its source or this script."
    rm -f "${OUTPUT}"
  else
    echo "BuildVoxelGpu.sh: nvcc not found; skipping the voxel sensor's GPU path."
  fi
  exit 0
fi

CUDA_LIB=$(dirname "$(dirname "${NVCC}")")/lib64

# One host compiler for nvcc and the link, so the static C++ library matches
# the code that uses it. CARLA's default g++ may be too old for nvcc.
HOST_CXX=${VOXEL_GPU_CXX:-}

for CANDIDATE in g++-11 g++-12 g++-13 g++; do
  if [ -z "${HOST_CXX}" ] && command -v "${CANDIDATE}" > /dev/null; then
    HOST_CXX=$(command -v "${CANDIDATE}")
  fi
done

mkdir -p "${BUILD}" "$(dirname "${OUTPUT}")"

echo "BuildVoxelGpu.sh: building $(basename "${OUTPUT}") with ${NVCC}."

# Only a C interface is exported, and the CUDA runtime and C++ standard
# library are linked statically, so the library does not clash with the
# engine's own runtime.
"${NVCC}" -ccbin "${HOST_CXX}" -O3 -std=c++17 -arch=all-major -Xcompiler -fPIC,-fvisibility=hidden \
  -c "${SOURCE}" -o "${BUILD}/VoxelGpu.o"

"${HOST_CXX}" -shared -o "${OUTPUT}" "${BUILD}/VoxelGpu.o" \
  -L"${CUDA_LIB}" -lcudart_static -ldl -lrt -lpthread \
  -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL -Wl,-z,defs

echo "BuildVoxelGpu.sh: done."
