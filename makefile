# Path to an MFEM build (build directory or install prefix containing
# config/config.mk).  Override on the command line, e.g.
#    make MFEM_DIR=/path/to/mfem MFEM_LIB_SUFFIX=
MFEM_DIR ?= ../mfem_moose_custOp/framework/contrib/mfem/build-opt
###MFEM_DIR=../../MOOSE_BUILDS/cust_ops/framework/contrib/mfem/build-opt

# MOOSE builds of MFEM name the library libmfem-opt; set to empty for a
# standard MFEM build.
MFEM_LIB_SUFFIX ?= -opt

CONFIG_MK = $(MFEM_DIR)/config/config.mk
include $(CONFIG_MK)

# BUILD=release (default: MFEM's optimisation flags) or BUILD=debug
# (-g -O0 with AddressSanitizer).  Use release builds for any timing or
# iteration-count study.
BUILD ?= release

TARGET = lmmhd_solver

SRC = lmmhd_solver.cpp \
      src/VectorConvectionIntegrator.cpp \
      src/LmmhdOperator.cpp \
      src/constants.cpp \
      src/tools.cpp

OBJ = $(SRC:.cpp=.o)

TEST_TARGETS = tests/test_convection_skew
TEST_OBJ = tests/test_convection_skew.o src/VectorConvectionIntegrator.o

DEP = $(OBJ:.o=.d) $(TEST_OBJ:.o=.d)

CXX = $(MFEM_CXX)
# -MMD -MP: track header dependencies (header-only code such as
# LiPreconditioner.hpp and InputParser.hpp is otherwise never rebuilt).
CXXFLAGS = $(MFEM_FLAGS) -Iinclude -MMD -MP
LDLIBS = $(subst -lmfem,-lmfem$(MFEM_LIB_SUFFIX),$(MFEM_LIBS))

ifeq ($(BUILD),debug)
   CXXFLAGS += -g -O0 -fsanitize=address
   LDFLAGS += -fsanitize=address
endif

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

tests/test_convection_skew: $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

test: $(TEST_TARGETS)
	./tests/test_convection_skew

-include $(DEP)

clean:
	rm -f $(TARGET) $(TEST_TARGETS) $(OBJ) $(TEST_OBJ) $(DEP) *~ core

.PHONY: all clean test
