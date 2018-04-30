#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <vector>

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

/*
__global__ void compute_forces_gpu(particle_t * particles, int n)
{
  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particles[tid].ax = particles[tid].ay = 0;
  for(int j = 0 ; j < n ; j++)
    apply_force_gpu(particles[tid], particles[j]);

}
*/

__global__ void compute_forces_gpu( particle_t &particles, int n, int &bins, int &particlesInBin, double binsize ){

    /*
        "particles" contains actual particles(structures)
        "bins" contains index of particles sorted by Bin number
        "particlesInBin" contains number of particles in each bin; helps traverse "bins"
        "n" is the number of particles in "particles"
        "binsize" is maximum binsize in 1-D (geometric-distance-wise) = cutoff
    */

    // each block computes a bin using 8 neighbors in a common case

    // get current block from global to shared memory
    __shared__ particle_t currentBin[n/2];

    // get 8 neighboring blocks from global to shared memory
    __shared__ particle_t northBin[n/2];
    __shared__ particle_t southBin[n/2];
    __shared__ particle_t eastBin[n/2];
    __shared__ particle_t westBin[n/2];
    __shared__ particle_t northEastBin[n/2];
    __shared__ particle_t northWestBin[n/2];
    __shared__ particle_t southEastBin[n/2];
    __shared__ particle_t southWestBin[n/2];

    // each thread handles a particle
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // we are not at an edge or a corner. -- Most common case
    if( (Location.Left | Location.Right | Location.Top | Location.Bottom) == false){
        
    }
    // Top Row
    else if( Location.Top ){
        // most common case for the top row -- Not in a corner.
        if( (Location.Left | Location.Right) == false){
                    //printf("Top Row %d \n", BinIndex );
 
        }
        else if( (!Location.Left) && Location.Right ){
                    //printf("Top Row Right %d \n", BinIndex );
                    // Right == East

        }
        else if ( Location.Left && (!Location.Right) ){
                    //printf("Top Row Left %d \n", BinIndex );

        }

    }
    else if( Location.Bottom ){
        // most common case for the top row -- Not in a corner.
        if((Location.Left | Location.Right) == false){
                    //printf("Bottom Row %d \n", BinIndex );

        }
        else if( (!Location.Left) && Location.Right ){
                    // Right == East
                    //printf("Bottom Row Right %d \n", BinIndex );

        }
        else if( Location.Left && (!Location.Right) ){
                    //printf("Bottom Row Left %d ", BinIndex );

        }

    }
    else if(Location.Left){
                //printf("Left %d \n", BinIndex );

    }
    else if(Location.Right){
                //printf("Right %d \n", BinIndex );

    }
    else{
        printf("Getting another bin case\n");
    }

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

        for(int particleIndex = 0; particleIndex < n; ++particleIndex){

            // place particles in bins
            int BinX = (int)(particles[particleIndex].x/binsize);
            int BinY = (int)(particles[particleIndex].y/binsize);

            int BinNum = BinX + NumofBinsEachSide*BinY;
            
            Bins[BinNum].push_back(particleIndex);
        }

        /// Beginning conversion from STL to primitives ///

        // Find number of particles in each bin
        int* particlesInBin = (int*) malloc( sizeof(int) * NumofBins );
        int prev_p=0;
        for( int b = 0; b < NumofBins; ++b){
            particlesInBin[b] = Bins[b].size();
            //totalSize += particlesInBin[b];

            int p;
            // prepare bins array from 2d Bins vector
            for( p=0; p < particlesInBin[b]; ++p ){
                bins[prev_p + p] = Bins[b].at(p);
            }
            prev_p += p;
        }
        int totalSize = prev_p;

        void* d_pib;
        cudaMalloc( &d_pib, sizeof(int) * NumofBins );
        int* d_particlesInBin = d_pib;

        cudaMemcpy( d_particlesInBin, particlesInBin, sizeof(int) * NumofBins, cudaMemcpyHostToDevice );

        void* B;
        cudaMalloc( &B, sizeof(int) * totalSize );
        int* d_bins = B;

        cudaMemcpy( d_bins, bins, sizeof(int) * totalSize, cudaMemcpyHostToDevice );

        //int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
	compute_forces_gpu <<< NumofBins, NUM_THREADS >>> (d_particles, n, d_bins, d_particlesInBin, binsize);
        
        //
        //  move particles
        //
	move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
        
        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
	    // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
	}
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
