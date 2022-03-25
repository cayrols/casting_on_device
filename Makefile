include Makefile.in

CUDASRC = \
		casting_interface.cu 

LIBNAME = compress_iface

OBJ=$(addprefix $(BUILD)/, $(SRC:.c=.o)) $(addprefix $(BUILD)/, $(CUDASRC:.cu=.o))

.PHONY: lib lib_shared clean gen_tags install

all:
	@if [ ! -d $(BUILD) ]; then mkdir -v $(BUILD); fi
	@echo $(OBJ)
	make obj

CUDA_INC ?= ${CUDA_DIR}/include
CUDA_LIB ?= ${CUDA_DIR}/lib64

obj: $(OBJ) 

lib:
	@if [ ! -d $(BUILD)/$(LIB) ]; then mkdir -pv $(BUILD)/$(LIB); fi
	ar vr $(BUILD)/$(LIB)/lib$(LIBNAME).a $(OBJ)
	ranlib $(BUILD)/$(LIB)/lib$(LIBNAME).a

lib_shared:
	@if [ ! -d $(BUILD)/$(LIB) ]; then mkdir -pv $(BUILD)/$(LIB); fi
	$(CC) -shared -fPIC -o $(BUILD)/$(LIB)/lib$(LIBNAME).so $(OBJ)

INC=-I . -I .. -I ${CUDA_INC}

$(BUILD)/%.o : %.c
	$(CC) $(OPTIONS) -o $@ -c $< $(INC) $(CFLAGS) -MMD

$(BUILD)/%.o : %.cu
	$(NVCC) $(OPTIONS) $(NVCCFLAGS) -o $@ -c $< $(INC)

gen_ctags:
	@echo "Create $(BUILD)/mpi_packing.tags"
	-@ctags --file-scope=no -R -o $(BUILD)/mpi_packing.tags `pwd`

install:
	@if [ ! -d $(INSTALL) ]; then mkdir -v $(INSTALL); fi
	@if [ ! -d $(INSTALL)/bin ]; then mkdir -v $(INSTALL)/bin; fi
	@if [ ! -d $(INSTALL)/include ]; then mkdir -v $(INSTALL)/include; fi
#cp -v $(SRC:.c=.h) $(CUDASRC:.cu=.cuh) $(INSTALL)/include
	cp -v *.h *.cuh $(INSTALL)/include
	@if [ ! -d $(INSTALL)/$(LIB) ]; then mkdir -v $(INSTALL)/$(LIB); fi
	cp -v $(BUILD)/$(LIB)/lib$(LIBNAME).a $(INSTALL)/$(LIB)

clean:
	-$(RM) $(OBJ)
	-$(RM) $(BUILD)/$(LIB)/lib$(LIBNAME).a
