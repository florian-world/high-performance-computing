#include <mpi.h>
#include <stdio.h>
#include <cmath>


struct particle
{
  int id;
  double x[3];
  bool state;
  double gamma;
};



int main(int argc, char ** argv)
{
	MPI_Init(&argc, &argv);

  int rank,size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  
  particle p;

  MPI_Datatype MPI_PARTICLE;

  if (rank == 0)
  {
    p.id = 1;
    p.x[0] = 3.14159265359;
    p.x[1] = 2.71828182846;
    p.x[2] = ( 1.0 + sqrt(5.0) ) / 2.0;
    p.state = false;
    p.gamma = 0.57721566490;
    
    //MPI_Send(&p,1,MPI_PARTICLE,1,100,MPI_COMM_WORLD);
  }
  else if (rank == 1)
  {
    //MPI_Recv(&p,1,MPI_PARTICLE,0,100,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
   
    printf("%d \n", p.id );
    printf("%10.8f \n", p.x[0]  );
    printf("%10.8f \n", p.x[1]  );
    printf("%10.8f \n", p.x[2]  );
    printf("%d \n"    , p.state );
    printf("%10.8f \n", p.gamma );
  }
  else
  {
    printf("Run with exactly two ranks!\n");
    int err = 1;
    MPI_Abort(MPI_COMM_WORLD,err);
  }

  MPI_Finalize();
  return 0;
}