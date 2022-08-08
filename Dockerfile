FROM ubuntu:20.04

# Install OS support packages
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends -o=Dpkg::Use-Pty=0 \
        build-essential \
        ca-certificates \
	cmake \
        gnupg \
        libarchive13 \
        m4 \
        pkg-config \
        zlib1g \
        zlib1g-dev

# Add apt repository public key
ARG url=https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
ADD $url /
RUN file=$(basename "$url") && \
    apt-key add "$file" && \
    rm "$file"

# Configure the apt repository
ARG repo=https://apt.repos.intel.com/oneapi
RUN echo "deb $repo all main" > /etc/apt/sources.list.d/oneAPI.list

# Install Intel oneapi packages
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends -o=Dpkg::Use-Pty=0 \
        intel-oneapi-dev-utilities \
        intel-oneapi-mpi-devel \
        intel-oneapi-openmp \
        intel-oneapi-compiler-fortran \
        intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic

# Set /opt/intel/oneapi/setvars.sh variables
ENV ACL_BOARD_VENDOR_PATH=/opt/Intel/OpenCLFPGA/oneAPI/Boards
ENV CLASSPATH=/opt/intel/oneapi/mpi/2021.2.0//lib/mpi.jar
ENV CMAKE_PREFIX_PATH=/opt/intel/oneapi/tbb/2021.2.0/env/..:
ENV CPATH=/opt/intel/oneapi/tbb/2021.2.0/env/../include:/opt/intel/oneapi/mpi/2021.2.0//include:/opt/intel/oneapi/dev-utilities/2021.2.0/include:/opt/intel/oneapi/compiler/2021.2.0/linux/include
ENV FI_PROVIDER_PATH=/opt/intel/oneapi/mpi/2021.2.0//libfabric/lib/prov:/usr/lib64/libfabric
ENV INFOPATH=/opt/intel/oneapi/debugger/10.1.1/documentation/info/
ENV INTELFPGAOCLSDKROOT=/opt/intel/oneapi/compiler/2021.2.0/linux/lib/oclfpga
ENV INTEL_PYTHONHOME=/opt/intel/oneapi/debugger/10.1.1/dep
ENV I_MPI_ROOT=/opt/intel/oneapi/mpi/2021.2.0
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/tbb/2021.2.0/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.2.0//libfabric/lib:/opt/intel/oneapi/mpi/2021.2.0//lib/release:/opt/intel/oneapi/mpi/2021.2.0//lib:/opt/intel/oneapi/debugger/10.1.1/dep/lib:/opt/intel/oneapi/debugger/10.1.1/libipt/intel64/lib:/opt/intel/oneapi/debugger/10.1.1/gdb/intel64/lib:/opt/intel/oneapi/compiler/2021.2.0/linux/lib:/opt/intel/oneapi/compiler/2021.2.0/linux/lib/x64:/opt/intel/oneapi/compiler/2021.2.0/linux/lib/emu:/opt/intel/oneapi/compiler/2021.2.0/linux/lib/oclfpga/host/linux64/lib:/opt/intel/oneapi/compiler/2021.2.0/linux/lib/oclfpga/linux64/lib:/opt/intel/oneapi/compiler/2021.2.0/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/compiler/2021.2.0/linux/compiler/lib
ENV LIBRARY_PATH=/opt/intel/oneapi/tbb/2021.2.0/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.2.0//libfabric/lib:/opt/intel/oneapi/mpi/2021.2.0//lib/release:/opt/intel/oneapi/mpi/2021.2.0//lib:/opt/intel/oneapi/compiler/2021.2.0/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/compiler/2021.2.0/linux/lib
ENV MANPATH=/opt/intel/oneapi/mpi/2021.2.0/man:/opt/intel/oneapi/debugger/10.1.1/documentation/man::/opt/intel/oneapi/compiler/2021.2.0/documentation/en/man/common::
ENV OCL_ICD_FILENAMES=libintelocl_emu.so:libalteracl.so:/opt/intel/oneapi/compiler/2021.2.0/linux/lib/x64/libintelocl.so
ENV ONEAPI_ROOT=/opt/intel/oneapi
ENV PATH=/opt/intel/oneapi/mpi/2021.2.0//libfabric/bin:/opt/intel/oneapi/mpi/2021.2.0//bin:/opt/intel/oneapi/dev-utilities/2021.2.0/bin:/opt/intel/oneapi/debugger/10.1.1/gdb/intel64/bin:/opt/intel/oneapi/compiler/2021.2.0/linux/lib/oclfpga/llvm/aocl-bin:/opt/intel/oneapi/compiler/2021.2.0/linux/lib/oclfpga/bin:/opt/intel/oneapi/compiler/2021.2.0/linux/bin/intel64:/opt/intel/oneapi/compiler/2021.2.0/linux/bin:/opt/intel/oneapi/compiler/2021.2.0/linux/ioc/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV SETVARS_COMPLETED=1
ENV SETVARS_VARS_PATH=/opt/intel/oneapi/tbb/latest/env/vars.sh
ENV TBBROOT=/opt/intel/oneapi/tbb/2021.2.0/env/..

# Set compilers to MPI wrappers
ENV CC='mpiicc'
ENV FC='mpiifort'
ENV CXX='mpiicpc'

# Install HDF5
ARG hdf5_url=https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.7/src/hdf5-1.10.7.tar.gz
ADD $hdf5_url /
RUN file=$(basename "$hdf5_url") && \
    tar -xzvf $file && \
    cd hdf5-1.10.7 && \
    ./configure --prefix=/opt/hdf5 --enable-parallel --disable-tools --disable-fortran --disable-cxx && \
    make -j2 && \
    make install && \
    cd .. && \
    rm -f "$file" && \
    rm -rf hdf5-1.10.7

# Install NetCDF-C
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hdf5/lib"
ENV CPPFLAGS="-I/opt/hdf5/include"
ENV LDFLAGS="-L/opt/hdf5/lib"
ARG netcdf_c_url=https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-c-4.7.4.tar.gz
ADD $netcdf_c_url /
RUN file=$(basename "$netcdf_c_url") && \
    tar -xzvf $file && \
    cd netcdf-c-4.7.4 && \
    ./configure --prefix=/opt/netcdf --disable-dap --disable-utilities && \
    make -j2 && \
    make install && \
    cd .. && \
    rm -f "$file" && \
    rm -rf netcdf-c-4.7.4

# Install NetCDF-Fortran
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/netcdf/lib"
ENV CPPFLAGS="${CPPFLAGS} -I/opt/netcdf/include"
ENV LDFLAGS="${LDFLAGS} -L/opt/netcdf/lib"
ARG netcdf_fortran_url=https://github.com/Unidata/netcdf-fortran/archive/v4.5.3.tar.gz
ADD $netcdf_fortran_url /
RUN file=$(basename "$netcdf_fortran_url") && \
    tar -xzvf $file && \
    cd netcdf-fortran-4.5.3 && \
    ./configure --prefix=/opt/netcdf && \
    make -j2 && \
    make install && \
    cd .. && \
    rm -f "$file" && \
    rm -rf netcdf-fortran-4.5.3

# Add HDF5 and NetCDF to CMAKE_PREFIX_PATH
ENV CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/hdf5:/opt/netcdf