# Simple makefile

EXES := bin/lst_cpu bin/lst_cuda

SOURCES=$(wildcard code/core/*.cc)
OBJECTS_CPU=$(SOURCES:.cc=_cpu.o)
OBJECTS_CUDA=$(SOURCES:.cc=_cuda.o)
OBJECTS_ROCM=$(SOURCES:.cc=_rocm.o)
OBJECTS=$(OBJECTS_CPU) $(OBJECTS_CUDA) $(OBJECTS_ROCM)

CXX         = g++
CXXFLAGS    = -g -O2 -Wall -fPIC -Woverloaded-virtual -Wno-unused-function -fno-var-tracking -std=c++20
INCLUDEFLAGS= -ILST -I$(shell pwd) -Icode -Icode/core -I${ALPAKA_ROOT}/include -I/${BOOST_ROOT}/include $(shell rooutil-config --include) -I$(shell root-config --incdir) -I${TRACKLOOPERDIR}/../../../ -I${CMSSW_BASE}/src -I${FMT_ROOT}/include -I../interface/ -I../interface/alpaka/ -I../src/ -I../src/alpaka/
ifdef CMSSW_RELEASE_BASE
INCLUDEFLAGS:= ${INCLUDEFLAGS} -I${CMSSW_RELEASE_BASE}/src
endif
LDFLAGS     = -g -O2 $(LSTLIB) -L${TRACKLOOPERDIR}/LST $(shell rooutil-config --libs) $(shell root-config --libs)
LDFLAGS_CUDA= -L${CUDA_HOME}/lib64 -lcudart
LDFLAGS_ROCM= -L${ROCM_ROOT}/lib -lamdhip64
ALPAKAFLAGS = -DALPAKA_DEBUG=0
CUDAINCLUDE = -I${CUDA_HOME}/include
ROCMINCLUDE = -I${ROCM_ROOT}/include
ALPAKA_CPU  = -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_DISABLE_VENDOR_RNG -DALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
ALPAKA_CUDA = -DALPAKA_ACC_GPU_CUDA_ENABLED -DALPAKA_HOST_ONLY -DALPAKA_DISABLE_VENDOR_RNG -DALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
ALPAKA_ROCM = -DALPAKA_ACC_GPU_HIP_ENABLED -DALPAKA_HOST_ONLY -DALPAKA_DISABLE_VENDOR_RNG -D__HIP_PLATFORM_HCC__ -D__HIP_PLATFORM_AMD__ -DALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
EXTRAFLAGS  = -ITMultiDrawTreePlayer -Wunused-variable -lTMVA -lEG -lGenVector -lXMLIO -lMLP -lTreePlayer -fopenmp -DLST_STANDALONE
DOQUINTUPLET =
CUTVALUEFLAG =
CUTVALUEFLAG_FLAGS = -DCUT_VALUE_DEBUG

PRIMITIVEFLAG =
PRIMITIVEFLAG_FLAGS = -DPRIMITIVE_STUDY

# Optional transformer model inference with libtorch (TRANSFORMER=1, i.e. lst_make_tracklooper -T).
# Only the lst binaries link libtorch; the LST library stays torch-free. Toggling requires a clean build (-m).
# RPATH (--disable-new-dtags) so that the libtorch found at build time is also the one used at run time.
TORCHFLAGS =
TORCHLDFLAGS =
TORCHLDFLAGS_CUDA =
ifeq ($(TRANSFORMER),1)
TORCHFLAGS = -DLST_TRANSFORMER -isystem ${TORCH_ROOT}/include -isystem ${TORCH_ROOT}/include/torch/csrc/api/include
TORCHLDFLAGS = -L${TORCH_ROOT}/lib -Wl,--disable-new-dtags -Wl,-rpath,${TORCH_ROOT}/lib -Wl,--no-as-needed -ltorch -ltorch_cpu -lc10 -Wl,--as-needed
TORCHLDFLAGS_CUDA = -Wl,--no-as-needed -ltorch_cuda -lc10_cuda -Wl,--as-needed
endif

all: rooutil efficiency $(EXES)

cutvalue: CUTVALUEFLAG = ${CUTVALUEFLAG_FLAGS}
cutvalue: rooutil efficiency $(EXES)

primitive: PRIMITIVEFLAG = ${PRIMITIVEFLAG_FLAGS}
primitive: rooutil efficiency $(EXES)

cutvalue_primitive: CUTVALUEFLAG = ${CUTVALUEFLAG_FLAGS}
cutvalue_primitive: PRIMITIVEFLAG = ${PRIMITIVEFLAG_FLAGS}
cutvalue_primitive: rooutil efficiency $(EXES)


bin/lst_cpu: LSTLIB=-llst_cpu
bin/lst_cpu: bin/lst_cpu.o $(OBJECTS_CPU)
	$(CXX) $(LDFLAGS) $(EXTRAFLAGS) $(INCLUDEFLAGS) $(ALPAKAFLAGS) $^ $(ROOTLIBS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(DOQUINTUPLET) $(ALPAKA_CPU) $(TORCHLDFLAGS) -o $@
bin/lst_cuda: LSTLIB=-llst_cuda
bin/lst_cuda: bin/lst_cuda.o $(OBJECTS_CUDA)
	$(CXX) $(LDFLAGS) $(EXTRAFLAGS) $(INCLUDEFLAGS) $(ALPAKAFLAGS) $^ $(ROOTLIBS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(DOQUINTUPLET) $(ALPAKA_CUDA) $(LDFLAGS_CUDA) $(TORCHLDFLAGS) $(TORCHLDFLAGS_CUDA) -o $@
bin/lst_rocm: LSTLIB=-llst_rocm
bin/lst_rocm: bin/lst_rocm.o $(OBJECTS_ROCM)
	$(CXX) $(LDFLAGS) $(EXTRAFLAGS) $(INCLUDEFLAGS) $(ALPAKAFLAGS) $^ $(ROOTLIBS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(DOQUINTUPLET) $(ALPAKA_ROCM) $(LDFLAGS_ROCM) $(TORCHLDFLAGS) -o $@

%_cpu.o: %.cc rooutil
	$(CXX) $(CXXFLAGS) $(EXTRAFLAGS) $(INCLUDEFLAGS) $(ALPAKAFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(DOQUINTUPLET) $(ALPAKA_CPU) $(TORCHFLAGS) $< -c -o $@
%_cuda.o: %.cc rooutil
	$(CXX) $(CXXFLAGS) $(EXTRAFLAGS) $(INCLUDEFLAGS) $(ALPAKAFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(DOQUINTUPLET) $(ALPAKA_CUDA) $(CUDAINCLUDE) $(TORCHFLAGS) $< -c -o $@
%_rocm.o: %.cc rooutil
	$(CXX) $(CXXFLAGS) $(EXTRAFLAGS) $(INCLUDEFLAGS) $(ALPAKAFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(DOQUINTUPLET) $(ALPAKA_ROCM) $(ROCMINCLUDE) $(TORCHFLAGS) $< -c -o $@

rooutil:
	$(MAKE) -C code/rooutil/

efficiency: rooutil
	$(MAKE) -C efficiency/

clean:
	rm -f $(OBJECTS) bin/*.o $(EXES) bin/lst
	rm -f code/rooutil/*.so code/rooutil/*.o
	rm -f bin/lst.o
	rm -f LST/*.o
	cd efficiency/ && make clean

.PHONY: rooutil efficiency
