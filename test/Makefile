#################################################################################################
#  Build script for unit tests
#  Usage: make <target> [sm=sm21]
#################################################################################################

NVCC = "$(shell which nvcc)"

#------------------------------------------------------------------------------------------------
#  Target architecture
#------------------------------------------------------------------------------------------------

DEVICE_ARCH_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\"
DEVICE_ARCH_SM30 = -gencode=arch=compute_30,code=\"sm_30,compute_30\"
DEVICE_ARCH_SM21 = -gencode=arch=compute_20,code=\"sm_21,compute_20\"
DEVICE_ARCH_SM20 = -gencode=arch=compute_20,code=\"sm_20,compute_20\"
DEVICE_ARCH_SM13 = -gencode=arch=compute_13,code=\"sm_13,compute_13\"
DEVICE_ARCH_SM11 = -gencode=arch=compute_11,code=\"sm_11,compute_11\"
DEVICE_ARCH_SM10 = -gencode=arch=compute_10,code=\"sm_10,compute_10\"

ifdef sm
	SM_ARCH = $(sm)
else
	SM_ARCH = sm21
endif

ifeq (sm35, $(findstring sm35, $(SM_ARCH)))
	DEVICE_ARCH = $(DEVICE_ARCH_SM35)
endif
ifeq (sm30, $(findstring sm30, $(SM_ARCH)))
	DEVICE_ARCH = $(DEVICE_ARCH_SM30)
endif
ifeq (sm21, $(findstring sm21, $(SM_ARCH)))
	DEVICE_ARCH = $(DEVICE_ARCH_SM21)
endif
ifeq (sm20, $(findstring sm20, $(SM_ARCH)))
	DEVICE_ARCH = $(DEVICE_ARCH_SM20)
endif
ifeq (sm13, $(findstring sm13, $(SM_ARCH)))
	DEVICE_ARCH = $(DEVICE_ARCH_SM13)
endif
ifeq (sm11, $(findstring sm11, $(SM_ARCH)))
	DEVICE_ARCH = $(DEVICE_ARCH_SM11)
endif
ifeq (sm10, $(findstring sm10, $(SM_ARCH)))
	DEVICE_ARCH = $(DEVICE_ARCH_SM10)
endif

#------------------------------------------------------------------------------------------------
#  Includes
#------------------------------------------------------------------------------------------------

INC = -I. -I../gcuda

#------------------------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------------------------

LIB = -lgtest

#------------------------------------------------------------------------------------------------
#  Defines
#------------------------------------------------------------------------------------------------

DEFINES =

#------------------------------------------------------------------------------------------------
#  Compiler flags
#------------------------------------------------------------------------------------------------

HOST_ARCH = 

NVCCFLAGS = -Xptxas -v -Xcudafe -\#

#------------------------------------------------------------------------------------------------
#  Dependency list
#------------------------------------------------------------------------------------------------

DEPS =

#------------------------------------------------------------------------------------------------
#  make all
#------------------------------------------------------------------------------------------------

all:
	make compare
	make hostarray
	make hostvector
	make devicearray
	make devicevector

#------------------------------------------------------------------------------------------------
#  make clean
#------------------------------------------------------------------------------------------------

clean:
	rm -f bin/*
	rm -f *.i* *.cubin *.ptx *.o

#------------------------------------------------------------------------------------------------
#  make compare
#------------------------------------------------------------------------------------------------

compare: bin/compare

bin/compare : compare.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(DEVICE_ARCH) -o bin/compare compare.cu $(NVCCFLAGS) $(HOST_ARCH) $(INC) $(LIB) -O3

#------------------------------------------------------------------------------------------------
#  make hostarray
#------------------------------------------------------------------------------------------------

hostarray: bin/hostarray

bin/hostarray : hostarray.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(DEVICE_ARCH) -o bin/hostarray hostarray.cu $(NVCCFLAGS) $(HOST_ARCH) $(INC) $(LIB) -O3
	
#------------------------------------------------------------------------------------------------
#  make hostvector
#------------------------------------------------------------------------------------------------

hostvector: bin/hostvector

bin/hostvector : hostvector.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(DEVICE_ARCH) -o bin/hostvector hostvector.cu $(NVCCFLAGS) $(HOST_ARCH) $(INC) $(LIB) -O3

#------------------------------------------------------------------------------------------------
#  make devicearray
#------------------------------------------------------------------------------------------------

devicearray: bin/devicearray

bin/devicearray : devicearray.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(DEVICE_ARCH) -o bin/devicearray devicearray.cu $(NVCCFLAGS) $(HOST_ARCH) $(INC) $(LIB) -O3

#------------------------------------------------------------------------------------------------
#  make devicevector
#------------------------------------------------------------------------------------------------

devicevector: bin/devicevector

bin/devicevector : devicevector.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(DEVICE_ARCH) -o bin/devicevector devicevector.cu $(NVCCFLAGS) $(HOST_ARCH) $(INC) $(LIB) -O3

