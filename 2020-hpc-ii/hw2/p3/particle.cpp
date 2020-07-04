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

  MPI_Aint p_lb, p_id, p_x, p_state, p_gamma, p_ub;

  MPI_Get_address(&p, &p_lb);
  MPI_Get_address(&p.id, &p_id);
  MPI_Get_address(&p.x, &p_x);
  MPI_Get_address(&p.state, &p_state);
  MPI_Get_address(&p.gamma, &p_gamma);
  MPI_Get_address(&p + 1, &p_ub);

  MPI_Datatype MPI_PARTICLE;
  int blocklens[] = {0, 1, 3, 1, 1, 0};
  MPI_Aint offsets[] = {0, p_id - p_lb, p_x - p_lb, p_state - p_lb, p_gamma - p_lb, p_ub - p_lb};
  MPI_Datatype types[] = {MPI_LB, MPI_INT, MPI_DOUBLE, MPI_CXX_BOOL, MPI_DOUBLE, MPI_UB};
  MPI_Type_create_struct(6, blocklens, offsets, types, &MPI_PARTICLE);
  MPI_Type_commit(&MPI_PARTICLE);

  if (rank == 0)
  {
    p.id = 1;
    p.x[0] = 3.14159265359;
    p.x[1] = 2.71828182846;
    p.x[2] = ( 1.0 + sqrt(5.0) ) / 2.0;
    p.state = false;
    p.gamma = 0.57721566490;
    
    MPI_Send(&p,1,MPI_PARTICLE,1,100,MPI_COMM_WORLD);
  }
  else if (rank == 1)
  {
    MPI_Recv(&p,1,MPI_PARTICLE,0,100,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
   
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
  MPI_Type_free(&MPI_PARTICLE);
  MPI_Finalize();
  return 0;
}