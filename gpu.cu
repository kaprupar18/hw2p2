#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <set>

#include "common.h"

#define NUM_THREADS 1024

extern double size;
//
//  benchmarking program
//

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}

__device__ Bin_Location_t getBinLocation(const int BinIndex, const int NumofBinsEachSide, const int NumofBins )
{ // assumes square geomerty!

    Bin_Location_t Temp; 

    Temp.Left = ((BinIndex%NumofBinsEachSide) == 0) ? true : false;
    Temp.Right = ((BinIndex%NumofBinsEachSide) == (NumofBinsEachSide-1) ) ? true : false;
    Temp.Top =  ((BinIndex < NumofBinsEachSide) )? true : false;
    Temp.Bottom = ((BinIndex > (NumofBins - NumofBinsEachSide - 1) ) )? true : false;

    return Temp; 
}

__global__ void compute_forces_gpu( particle_t* particles, int* bins, int* binHasParticles, double binsize, int NumofBinsEachSide ){

    //
    //  "particles" contains actual particles(structures)
    //  "bins" contains index of particles sorted by Bin number
    //  "binHasParticles" is true if currentBin has particle(s)
    //  "n" is the number of particles in "particles"
    //  "binsize" is maximum binsize in 1-D (geometric-distance-wise) = cutoff
    //

    // each thread computes each bin(particle)

    // each block computes a bin using 8 neighbors in a common case

    // get current block from global to shared memory
    __shared__ particle_t currentBin;
    __shared__ particle_t neighbors[8];

    // each thread handles a particle
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    //int tid = tidx * blockDim.x + tidy;

    if( tidx > NumofBinsEachSide-1 || tidy > NumofBinsEachSide-1 ){
        return;
    }

    // compute Bin indices...............CHECKED
    int currentBinId = tidx * blockDim.x + tidy;
    int nId = (tidx - 1) * blockDim.x + tidy;
    int sId = (tidx + 1) * blockDim.x + tidy;
    int eId = tidx * blockDim.x + tidy + 1;
    int wId = tidx * blockDim.x + tidy - 1;
    int neId = (tidx - 1) * blockDim.x + tidy + 1;
    int nwId = (tidx - 1) * blockDim.x + tidy - 1;
    int seId = (tidx + 1) * blockDim.x + tidy + 1;
    int swId = (tidx + 1) * blockDim.x + tidy - 1;

    __syncthreads();

    // Return if this bin is empty
    if(!binHasParticles[currentBinId]){
        return;
    }

    // get Bin Location on grid
    Bin_Location_t Location = getBinLocation(currentBinId, NumofBinsEachSide, NumofBinsEachSide*NumofBinsEachSide);

    // get particles in current bin
    currentBin = particles[bins[currentBinId]];

    particle_t nullParticle;
    nullParticle.x = nullParticle.y = nullParticle.vx = nullParticle.vy = nullParticle.ax = nullParticle.ay = -1;
    
    __syncthreads();

    // we are not at an edge or a corner. -- Most common case
    if( (Location.Left | Location.Right | Location.Top | Location.Bottom) == false){
        // get 8 neighbors
        neighbors[0] = particles[bins[nId]];
        neighbors[1] = particles[bins[sId]];
        neighbors[2] = particles[bins[eId]];
        neighbors[3] = particles[bins[wId]];
        neighbors[4] = particles[bins[neId]];
        neighbors[5] = particles[bins[seId]];
        neighbors[6] = particles[bins[swId]];
        neighbors[7] = particles[bins[nwId]];
    }
    // Top Row
    else if( Location.Top ){

        neighbors[0] = particles[bins[sId]];

        // most common case for the top row -- Not in a corner.
        if( (Location.Left | Location.Right) == false ){
            // get 4 bins: W, E, SW, SE
            neighbors[1] = particles[bins[eId]];
            neighbors[2] = particles[bins[wId]];
            neighbors[3] = particles[bins[seId]];
            neighbors[4] = particles[bins[swId]];
            
            neighbors[5] = nullParticle;
            neighbors[6] = nullParticle;
            neighbors[7] = nullParticle;
        }
        else if( (!Location.Left) && Location.Right ){
            // get 2 bins: W, SW
            neighbors[1] = particles[bins[wId]];
            neighbors[2] = particles[bins[swId]];

            neighbors[3] = nullParticle;
            neighbors[4] = nullParticle;
            neighbors[5] = nullParticle;
            neighbors[6] = nullParticle;
            neighbors[7] = nullParticle;
        }
        else if ( Location.Left && (!Location.Right) ){
            // get 2 bins: E, SE
            neighbors[1] = particles[bins[eId]];
            neighbors[2] = particles[bins[seId]];

            neighbors[3] = nullParticle;
            neighbors[4] = nullParticle;
            neighbors[5] = nullParticle;
            neighbors[6] = nullParticle;
            neighbors[7] = nullParticle;
        }
    }
    else if( Location.Bottom ){

        neighbors[0] = particles[bins[nId]];

        // most common case for the top row -- Not in a corner.
        if((Location.Left | Location.Right) == false){
            // get 4 bins: W, E, NW, NE
            neighbors[1] = particles[bins[wId]];
            neighbors[2] = particles[bins[eId]];
            neighbors[3] = particles[bins[nwId]];
            neighbors[4] = particles[bins[neId]];

            neighbors[5] = nullParticle;
            neighbors[6] = nullParticle;
            neighbors[7] = nullParticle;
        }
        else if( (!Location.Left) && Location.Right ){
            // get 2 bins: W, NW
            neighbors[1] = particles[bins[wId]];
            neighbors[2] = particles[bins[nwId]];

            neighbors[3] = nullParticle;
            neighbors[4] = nullParticle;
            neighbors[5] = nullParticle;
            neighbors[6] = nullParticle;
            neighbors[7] = nullParticle;
        }
        else if( Location.Left && (!Location.Right) ){
            // get 2 bins: E, NE
            neighbors[1] = particles[bins[eId]];
            neighbors[2] = particles[bins[neId]];

            neighbors[3] = nullParticle;
            neighbors[4] = nullParticle;
            neighbors[5] = nullParticle;
            neighbors[6] = nullParticle;
            neighbors[7] = nullParticle;
        }
    }
    else if(Location.Left){
        // get 5 bins: N, S, E, NE, SE
        neighbors[0] = particles[bins[nId]];
        neighbors[1] = particles[bins[sId]];
        neighbors[2] = particles[bins[eId]];
        neighbors[3] = particles[bins[neId]];
        neighbors[4] = particles[bins[seId]];

        neighbors[5] = nullParticle;
        neighbors[6] = nullParticle;
        neighbors[7] = nullParticle;
    }
    else if(Location.Right){
        // get 5 bins: N, S, W, NW, SW
        neighbors[0] = particles[bins[nId]];
        neighbors[1] = particles[bins[sId]];
        neighbors[2] = particles[bins[wId]];
        neighbors[3] = particles[bins[nwId]];
        neighbors[4] = particles[bins[swId]];

        neighbors[5] = nullParticle;
        neighbors[6] = nullParticle;
        neighbors[7] = nullParticle;
    }
    else{
        printf("Getting another bin case\n");
    }
    ///////////// Bins with particles READY! ///////////////////

    __syncthreads();
    
    for( int j = 0 ; j < 8 ; ++j ){
        if( neighbors[j].x == -1 )
            return;
        apply_force_gpu( currentBin, neighbors[j] );
        __syncthreads();
    }
    __syncthreads();
    
    // shared to global
    particles[bins[currentBinId]] = currentBin;

    __syncthreads();
}


__global__ void move_gpu (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    // once the force is applied awesome the accel is zero.
    p->ax = 0;
    p->ay = 0;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }

}



int main( int argc, char **argv )
{    
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize(); 

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );
    init_particles( n, particles );

    double size = getSize(); 
    int NumofBinsEachSide = getNumberofBins(size);
    int NumofBins = NumofBinsEachSide*NumofBinsEachSide;
    
    double binsize = getBinSize();

    std::vector< std::vector<int> > Bins(NumofBins, std::vector<int>(0));

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;

    int* particlesInBin = (int*) malloc( sizeof(int) * NumofBins );
    
    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
        // clear bins for the iteration
        for(int clear = 0; clear < Bins.size(); clear++ ){
            Bins[clear].clear();
        }
        //std::cout << NumofBins << " bins cleared" << std::endl;

        std::set<int> BinsWithParticles;
        for(int particleIndex = 0; particleIndex < n; ++particleIndex){

            // place particles in bins
            int BinX = (int)(particles[particleIndex].x/binsize);
            int BinY = (int)(particles[particleIndex].y/binsize);

            int BinNum = BinX + NumofBinsEachSide*BinY;
            
            Bins[BinNum].push_back(particleIndex);

            // store the bin which contain a particle. We will ignore the empty ones
            BinsWithParticles.insert(BinNum);
        }

        /// Beginning conversion from STL to primitives ///
        //std::cout << "Starting vector to linear array conversion" << std::endl;
        //std::cout << "BinsWithParticles: " << BinsWithParticles.size() << std::endl;
        // Find number of particles in each bin
        int prev_p=0;
        //particlesInBin[0] = Bins[0].size();
        int max = 0;

        // Linear array from of Bins[][]
        int* bins = (int*) malloc( sizeof(int) * n );
        int* binHasParticles = (int*) malloc( sizeof(int) * NumofBins );

        for( int b = 0; b < NumofBins; ++b){
            
            particlesInBin[b] = Bins[b].size();
            //std::cout << "NumofParticles in bin " << b << ": " << particlesInBin[b] << std::endl;
            max = particlesInBin[b] > max ? particlesInBin[b] : max;
            
            // binStartsAt is replaced by binsWithParticles since a bin has one particle or none
            if(particlesInBin[b] != 0){
                // bin has particle(s)
                binHasParticles[b] = true;    // store binId
            }
            else{
                binHasParticles[b] = false;
            }

            int p;
            // prepare bins array from 2d Bins vector
            for( p=0; p < particlesInBin[b]; ++p ){
                bins[prev_p + p] = Bins[b].at(p);
            }
            prev_p += p;

        }
        //std::cout << "Params: \nmax = " << max << "\ntotalSize = " << n << std::endl; 

        int* d_binHasParticles;
        cudaMalloc( (void**) &d_binHasParticles, sizeof(int) * NumofBins );
        cudaMemcpy( d_binHasParticles, binHasParticles, sizeof(int) * NumofBins, cudaMemcpyHostToDevice );

        int* d_bins;
        cudaMalloc( (void**) &d_bins, sizeof(int) * n );
        cudaMemcpy( d_bins, bins, sizeof(int) * n, cudaMemcpyHostToDevice );

        int val = (n<1000) ? 1 : (n/1000);
        dim3 blks(val);
        
        // v is NumofBinsEachSide rounded to the next power of 2
        int v = NumofBinsEachSide;
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        
        dim3 tds(v, v);

        cudaThreadSynchronize();
	compute_forces_gpu <<< blks, tds >>> (d_particles, d_bins, d_binHasParticles, binsize, NumofBinsEachSide);
        cudaThreadSynchronize();
        //
        //  move particles
        //
	move_gpu <<< blks, tds >>> (d_particles, n, size);
        cudaThreadSynchronize();
        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
	    // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
        }

        free(bins);
        free(binHasParticles);
        cudaFree(d_bins);
        cudaFree(d_binHasParticles);
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    free( particlesInBin );
    cudaFree(d_particles);

    if( fsave )
        fclose( fsave );
    
    return 0;
}
