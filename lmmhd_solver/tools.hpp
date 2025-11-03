#pragma once
#include "mfem.hpp"
#include <iostream>
using namespace mfem;
using namespace std;

// Print all parameter read in from input file.
void PrintParams(int nx, int ny, int nz, 
		real_t Lx, real_t Ly, real_t Lz,
                real_t clusterY, real_t clusterZ,
                real_t t_final, real_t dt, int vis_steps, 
		real_t Re, real_t Ha);

// Print finite element space sizes.
void PrintFESpaces(int j_space_size, int phi_space_size, 
        int v_space_size, int p_space_size);

// Clustering function (symmetric around center).
real_t cluster_symmetric(real_t xi, real_t factor);