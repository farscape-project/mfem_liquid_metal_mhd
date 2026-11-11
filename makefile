# Path to an MFEM build (build directory or install prefix containing
# config/config.mk).  Override on the command line, e.g.
#    make MFEM_DIR=/path/to/mfem MFEM_LIB_SUFFIX=
MFEM_DIR ?= ../mfem_moose_custOp/framework/contrib/mfem/build-opt
###MFEM_DIR=../../MOOSE_BUILDS/cust_ops/framework/contrib/mfem/build-opt
#MFEM_DIR ?= ../moose/framework/contrib/mfem/build-opt

# MOOSE builds of MFEM name the library libmfem-opt; set to empty for a
# standard MFEM build.
MFEM_LIB_SUFFIX ?= -opt

CONFIG_MK = $(MFEM_DIR)/config/config.mk
include $(CONFIG_MK)

# BUILD=release (default: MFEM's optimisation flags)
# BUILD=debug (-g -O0 with AddressSanitizer). 
BUILD ?= release

ifeq ($(BUILD),debug)
   CXXFLAGS += -g -O0 -fsanitize=address
   LDFLAGS += -fsanitize=address
endif

LIB    = liblmmhd.a
TARGET = lmmhd_solver

# Library: everything except main.  Add new src/*.cpp files here.
LIB_SRC = src/LmmhdTimeStepper.cpp \
          src/LmmhdOperator.cpp \
          src/VectorConvectionIntegrator.cpp \
          src/tools.cpp
LIB_OBJ = $(LIB_SRC:.cpp=.o)

APP_SRC = main.cpp
APP_OBJ = $(APP_SRC:.cpp=.o)

TEST_TARGETS = tests/test_convection_skew
TEST_OBJ = tests/test_convection_skew.o

DEP = $(LIB_OBJ:.o=.d) $(APP_OBJ:.o=.d) $(TEST_OBJ:.o=.d)

CXX = $(MFEM_CXX)
# -fPIC: the archive will be linked into MOOSE's shared library.
CXXFLAGS = $(MFEM_FLAGS) -Iinclude -fPIC -MMD -MP
LDLIBS = $(subst -lmfem,-lmfem$(MFEM_LIB_SUFFIX),$(MFEM_LIBS))

all: $(TARGET)
lib: $(LIB)

$(LIB): $(LIB_OBJ)
	$(AR) rcs $@ $^

$(TARGET): $(APP_OBJ) $(LIB)
	$(CXX) $(CXXFLAGS) $(APP_OBJ) -o $@ -L. -llmmhd $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

tests/test_convection_skew: $(TEST_OBJ) $(LIB)
	$(CXX) $(CXXFLAGS) $(TEST_OBJ) -o $@ -L. -llmmhd $(LDFLAGS) $(LDLIBS)

test: $(TEST_TARGETS)
	./tests/test_convection_skew

# Fails if any header leaks MFEM names into code that includes it.
check-headers:
	$(CXX) $(CXXFLAGS) -fsyntax-only tests/test_no_namespace_leak.cpp

-include $(DEP)

clean:
	rm -f $(TARGET) $(LIB) $(TEST_TARGETS) $(LIB_OBJ) $(APP_OBJ) $(TEST_OBJ) $(DEP) *~ core

.PHONY: all lib test check-headers clean
