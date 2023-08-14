#include <iostream>
#include <math.h>
#include <numeric>
#include <execution>
#include <vector>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <iomanip>
#include <omp.h>

#include "./functions.cpp"

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"block size", 1, NULL, 'b'},
  {"size", 1, NULL, 's'},
  {"iterations", 1, NULL, 'i'},
  {0,0,0,0}
};

int main(int argc, char* argv[]) {
    int n_row, n_col;
    n_row = n_col = 128; // deafult matrix size
    int opt, option_index=0;
    int block_size = 16;
    double *  A;
    double *  b, * x0;
    int iterations = 1;
    func_ret_t ret, ret1, ret2;

    while ((opt = getopt_long(argc, argv, "::s:b:i:", 
          long_options, &option_index)) != -1 ) {
    switch(opt){
      case 'b':
        block_size = atoi(optarg);
        break;
      case 's':
        n_col=n_row= atoi(optarg);
        break;
      case 'i':
        iterations = atoi(optarg);
        break;
      case '?':
        fprintf(stderr, "invalid option\n");
        break;
      case ':':
        fprintf(stderr, "missing argument\n");
        break;
      default:
        std::cout<<"Usage: "<< argv[0]<< "[-s matrix_size] \n" << std::endl;
        exit(EXIT_FAILURE);
        }
    }

    if ((optind < argc) || (optind == 1))
    {
        std::cout<<"Usage: "<< argv[0]<< "[-s matrix_size|-b blocksize <optional>]\n" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (n_row) 
    {
        printf("Creating matrix internally of size = %d\n", n_row);
        ret = create_matrix(&A, n_row);
        ret1 = create_vector(&b, n_row);
        if (ret != RET_SUCCESS && ret1 != RET_SUCCESS) 
        {
            A = NULL;
            std::cout<< stderr << "error creating matrix internally of size = "<< n_row << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    else 
    {
        printf("No input for matrix sise specified!\n");
        exit(EXIT_FAILURE);
    }


    std::cout << "Matrix size:  [" << n_row << "," << n_col << "]" <<std::endl;

    double* r = (double*) malloc(sizeof(double)*n_row);
    double* rp = (double*) malloc(sizeof(double)*n_row);

    double* p = (double*) malloc(sizeof(double)*n_row);

    double* alpha = (double*) malloc(sizeof(double)*1);
    double* beta = (double*) malloc(sizeof(double)*1);
    double* num = (double*) malloc(sizeof(double)*1); num[0] = 0.0;
    double* den = (double*) malloc(sizeof(double)*1); den[0] = 0.0;

    x0 = (double*)malloc(sizeof(double)*n_row);

    std::fill(x0,x0+n_row,0.0);

    { // omp scope

    
    auto N = static_cast<size_t>(n_row);

    auto kernel_start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < n_row; i++)
    {

        auto temp = 0.0;

        for (size_t j = 0; j < N; j++)
        {
        temp += A[i*N+j]*x0[j];
        }
        
        r[i] = b[i] - temp ;
    }

    for (size_t i = 0; i < N; i++)
    {
        p[i] = r[i];
    }

    for (size_t i = 0; i < n_row; i++)
    {
        rp[i] = r[i];
    }

    double err = 0.0;

    for (size_t i = 0; i < N; i++)
    {
        err += r[i]*r[i];
    }


    err = std::sqrt(err);

    double* accum = (double*) malloc(sizeof(double)*n_row);


    auto tolerance = 1E-5 ;

    while(err > tolerance)
    {

        std::fill(accum,accum+n_row,0.0);
        num[0] = 0.0;
        den[0] = 0.0;

        #pragma omp parallel for reduction(+:num[0])
        for (size_t i = 0; i < n_row; i++)
        {
            num[0] += r[i]*r[i];
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < n_row; i++)
        {
            for (size_t j = 0; j < n_col; j++)
            {
                accum[i] += p[i]*A[i*N+j]*p[j] ;
            }
            
        }
        
        den[0] = std::accumulate(accum, accum+n_row,0.0);
            
        alpha[0] = num[0] / den[0]; 

        #pragma omp parallel for
        for (size_t i = 0; i < n_row; i++)
        {
            x0[i] = alpha[0]*p[i];
        }

        #pragma omp parallel for
        for (size_t i = 0; i < n_row; i++)
        {
            double temp = 0.0;
            for (size_t j = 0; j < n_col; j++)
            {
                temp+= alpha[0]*A[i*N+j]*p[j];
            }
            r[i] = r[i] - temp;
            
        }

        err = 0.0;

        for (size_t i = 0; i < N; i++)
        {
            err += r[i]*r[i];
        }

        err = std::sqrt(err);

        
        
        if (err < tolerance)
        {
        break;
        }
        
        num[0] = 0.0;
        den[0] = 0.0;

        #pragma omp parallel for reduction(+:num[0])
        for (size_t i = 0; i < n_row; i++)
        {
            #pragma omp atomic
            num[0] += r[i]*r[i];
        }

        #pragma omp parallel for reduction(+:den[0])
        for (size_t i = 0; i < n_row; i++)
        {
            den[0] += rp[i]*rp[i];
        }

        beta[0] = num[0]/den[0];

        #pragma omp parallel for
        for (size_t i = 0; i < n_row; i++)
        {
            p[i] = r[i] + beta[0]*p[i];
        }

        for (size_t i = 0; i < n_row; i++)
        {
        rp[i] = r[i];
        }

    }


    auto kernel_end_time = std::chrono::high_resolution_clock::now();

    auto kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end_time - kernel_start_time);

    std::cout << "Average time taken to execute kernel : "<< kernel_duration.count()/(1E6) <<" seconds" <<std::endl;
    std::cout << "\n"; 
          
   
    }
  
    return 0;
    
}
