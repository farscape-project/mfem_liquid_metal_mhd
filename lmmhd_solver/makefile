MFEM_DIR ?= ../../..
CONFIG_MK = $(MFEM_DIR)/config/config.mk

include $(CONFIG_MK)

# Source files and executable
TARGET = lmmhd_solver
#SRC = lmmhd_solver.cpp VectorConvectionIntegrator.cpp LmmhdOperator.cpp constants.cpp BlockTriangularPreconditioner.cpp tools.cpp
#INC = VectorConvectionIntegrator.hpp LmmhdOperator.hpp constants.hpp LiPreconditioner.hpp BlockTriangularPreconditioner.hpp CrossProductMatrixCoefficient.hpp tools.hpp BoundaryConditions.hpp InputParser.hpp

# Source files
SRC = lmmhd_solver.cpp \
      src/VectorConvectionIntegrator.cpp \
      src/LmmhdOperator.cpp \
      src/constants.cpp \
      src/tools.cpp

# Header files (for reference only)
INC = include/VectorConvectionIntegrator.hpp \
      include/LmmhdOperator.hpp \
      include/constants.hpp \
      include/LiPreconditioner.hpp \
      include/CrossProductMatrixCoefficient.hpp \
      include/tools.hpp \
      include/BoundaryConditions.hpp \
      include/InputParser.hpp

# Compiler and flags
CXX = $(MFEM_CXX)
CXXFLAGS = $(MFEM_FLAGS) -Iinclude
LDFLAGS = $(MFEM_LIBS)

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)
	
clean:
	rm -f $(TARGET) *.o *~ core

.PHONY: all clean
