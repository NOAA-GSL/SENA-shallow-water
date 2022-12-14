FROM ubuntu:20.04

# Install OS support packages
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends -o=Dpkg::Use-Pty=0 \
        build-essential \
        ca-certificates \
        cmake \
        git \
        gnupg \
        libarchive13 \
        libboost-all-dev \
        libffi-dev \
        libgdbm-dev \
        libncurses5-dev \
        libnss3-dev \
        libsqlite3-dev \
        libssl-dev \
        libssl-dev \
        libbz2-dev \
        m4 \
        pkg-config \
        software-properties-common \
        wget \
        zlib1g \
        zlib1g-dev

# Install Python 3.9
ARG python_url=https://www.python.org/ftp/python/3.9.1/Python-3.9.1.tgz
ADD $python_url /
RUN file=$(basename "$python_url") && \
    tar -xzf $file && \
    cd Python-3.9.1 && \
    ./configure --enable-optimizations && \
    make -j2 && \
    make install && \
    cd .. && \
    rm -f "$file" && \
    rm -rf Python-3.9.1

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
ENV TBBROOT=/opt/intel/oneapi/tbb/2021.7.1/env/..
ENV ONEAPI_ROOT=/opt/intel/oneapi
ENV PKG_CONFIG_PATH=/opt/intel/oneapi/tbb/2021.7.1/env/../lib/pkgconfig:/opt/intel/oneapi/mpi/2021.7.1/lib/pkgconfig:/opt/intel/oneapi/compiler/2022.2.1/lib/pkgconfig
ENV FPGA_VARS_DIR=/opt/intel/oneapi/compiler/2022.2.1/linux/lib/oclfpga
ENV I_MPI_ROOT=/opt/intel/oneapi/mpi/2021.7.1
ENV FI_PROVIDER_PATH=/opt/intel/oneapi/mpi/2021.7.1//libfabric/lib/prov:/usr/lib64/libfabric
ENV DIAGUTIL_PATH=/opt/intel/oneapi/debugger/2021.7.1/sys_check/debugger_sys_check.py:/opt/intel/oneapi/compiler/2022.2.1/sys_check/sys_check.sh
ENV MANPATH=/opt/intel/oneapi/mpi/2021.7.1/man:/opt/intel/oneapi/debugger/2021.7.1/documentation/man:/opt/intel/oneapi/compiler/2022.2.1/documentation/en/man/common::
ENV GDB_INFO=/opt/intel/oneapi/debugger/2021.7.1/documentation/info/
ENV CMAKE_PREFIX_PATH=/opt/intel/oneapi/tbb/2021.7.1/env/..:/opt/intel/oneapi/compiler/2022.2.1/linux/IntelDPCPP
ENV CMPLR_ROOT=/opt/intel/oneapi/compiler/2022.2.1
ENV INFOPATH=/opt/intel/oneapi/debugger/2021.7.1/gdb/intel64/lib
ENV LIBRARY_PATH=/opt/intel/oneapi/tbb/2021.7.1/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.7.1//libfabric/lib:/opt/intel/oneapi/mpi/2021.7.1//lib/release:/opt/intel/oneapi/mpi/2021.7.1//lib:/opt/intel/oneapi/compiler/2022.2.1/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/compiler/2022.2.1/linux/lib
ENV OCL_ICD_FILENAMES=libintelocl_emu.so:libalteracl.so:/opt/intel/oneapi/compiler/2022.2.1/linux/lib/x64/libintelocl.so
ENV CLASSPATH=/opt/intel/oneapi/mpi/2021.7.1//lib/mpi.jar
ENV INTELFPGAOCLSDKROOT=/opt/intel/oneapi/compiler/2022.2.1/linux/lib/oclfpga
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/tbb/2021.7.1/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.7.1//libfabric/lib:/opt/intel/oneapi/mpi/2021.7.1//lib/release:/opt/intel/oneapi/mpi/2021.7.1//lib:/opt/intel/oneapi/debugger/2021.7.1/gdb/intel64/lib:/opt/intel/oneapi/debugger/2021.7.1/libipt/intel64/lib:/opt/intel/oneapi/debugger/2021.7.1/dep/lib:/opt/intel/oneapi/compiler/2022.2.1/linux/lib:/opt/intel/oneapi/compiler/2022.2.1/linux/lib/x64:/opt/intel/oneapi/compiler/2022.2.1/linux/lib/oclfpga/host/linux64/lib:/opt/intel/oneapi/compiler/2022.2.1/linux/compiler/lib/intel64_lin
ENV NLSPATH=/opt/intel/oneapi/compiler/2022.2.1/linux/compiler/lib/intel64_lin/locale/%l_%t/%N
ENV PATH=/opt/intel/oneapi/mpi/2021.7.1//libfabric/bin:/opt/intel/oneapi/mpi/2021.7.1//bin:/opt/intel/oneapi/dev-utilities/2021.7.1/bin:/opt/intel/oneapi/debugger/2021.7.1/gdb/intel64/bin:/opt/intel/oneapi/compiler/2022.2.1/linux/lib/oclfpga/bin:/opt/intel/oneapi/compiler/2022.2.1/linux/bin/intel64:/opt/intel/oneapi/compiler/2022.2.1/linux/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV INTEL_PYTHONHOME=/opt/intel/oneapi/debugger/2021.7.1/dep
ENV CPATH=/opt/intel/oneapi/tbb/2021.7.1/env/../include:/opt/intel/oneapi/mpi/2021.7.1//include:/opt/intel/oneapi/dev-utilities/2021.7.1/include

# Install Python dependencies
ADD ./requirements.txt .
RUN pip3 install -r ./requirements.txt && \
    rm -rf ./requirements.txt

# Set compilers to MPI wrappers
ENV CC='mpiicc'
ENV FC='mpiifort'
ENV CXX='mpiicpc'

## Install HDF5
ARG hdf5_url=https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.7/src/hdf5-1.10.7.tar.gz
ADD $hdf5_url /
RUN file=$(basename "$hdf5_url") && \
    tar -xzf $file && \
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
ARG netcdf_c_url=https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.7.4.tar.gz
ADD $netcdf_c_url /
RUN file=$(basename "$netcdf_c_url") && \
    tar -xzf $file && \
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
    tar -xzf $file && \
    cd netcdf-fortran-4.5.3 && \
    ./configure --prefix=/opt/netcdf && \
    make -j2 && \
    make install && \
    cd .. && \
    rm -f "$file" && \
    rm -rf netcdf-fortran-4.5.3

# Install Serialbox
ARG serialbox_url=https://github.com/GridTools/serialbox/archive/refs/tags/v2.6.1.tar.gz
ADD $serialbox_url /opt
RUN file=$(basename "$serialbox_url") && \
    cd /opt && \
    tar -xzf $file && \
    cd serialbox-2.6.1 && \
    mkdir build && \
    cd build && \
    cmake ../ -DSERIALBOX_ENABLE_FORTRAN=ON && \
    make && \
    make install && \
    cd .. && \
    rm -f "$file"

# Add HDF5 and NetCDF to CMAKE_PREFIX_PATH
ENV CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/hdf5:/opt/netcdf:/opt/serialbox-2.6.1/install
