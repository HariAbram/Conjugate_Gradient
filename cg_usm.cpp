#include <iostream>
#include <math.h>
#include <numeric>
#include <vector>
#include <CL/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <iomanip>

#include "./functions.cpp"

typedef std::chrono::duration<unsigned long long> my_duration;

using namespace cl;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"block size", 1, NULL, 'b'},
  {"size", 1, NULL, 's'},
  {"iterations", 1, NULL, 'i'},
  {"atomic", 1, NULL, 'a'},
  {0,0,0,0}
};

int main(int argc, char* argv[]) {
    int n_row, n_col;
    n_row = n_col = 128; // deafult matrix size
    int opt, option_index=0;
    int block_size = 16;
    double *  A;
    double *  b;
    int iterations = 5;
    func_ret_t ret, ret1, ret2;
    bool atomics = true;
    const char * atom = NULL;


    while ((opt = getopt_long(argc, argv, "::s:b:i:a:", 
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
      case 'a':
        atom = optarg;
        break;
      case '?':
        fprintf(stderr, "invalid option\n");
        break;
      case ':':
        fprintf(stderr, "missing argument\n");
        break;
      default:
        std::cout<<"Usage: "<< argv[0]<< "[-s matrix_size|-b blocksize <optional>| -a yes or no (for atomics)<optional>] \n" << std::endl;
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

    if (atom)
    {
        if (std::strcmp(atom, "no")) atomics = false;
    }
    


    std::cout << "Matrix size:  [" << n_row << "," << n_col << "]" <<std::endl;

    
    int k = 0;

    

    

    { // SYCL scope

    
    sycl::queue Q{};
    std::cout << "running on ..."<< std::endl;
    std::cout << Q.get_device().get_info<sycl::info::device::name>()<<"\n"<<std::endl;

    auto r = sycl::malloc_shared<double>(n_row*sizeof(double),Q); 
    auto rp = sycl::malloc_shared<double>(n_row*sizeof(double),Q); 

    auto p = sycl::malloc_shared<double>(n_row*sizeof(double),Q); 

    auto alpha = sycl::malloc_shared<double>(1*sizeof(double),Q); 
    alpha[0]=0.0;
    auto beta = sycl::malloc_shared<double>(1*sizeof(double),Q); 
    beta[0]=0.0;

    auto num = sycl::malloc_shared<double>(1*sizeof(double),Q);  
    num[0] = 0.0;
    auto den = sycl::malloc_shared<double>(1*sizeof(double),Q);  
    den[0] = 0.0;

    auto x0 = sycl::malloc_shared<double>(n_row*sizeof(double),Q);
    std::fill(x0,x0+n_row,0.0);

    auto A_shared = static_cast<double *>(sycl::malloc_shared(n_row*n_row*sizeof(double), Q));
    auto b_shared = static_cast<double *>(sycl::malloc_shared(n_row*sizeof(double), Q));

    //Q.memcpy(A_shared,A,sizeof(double)*n_row*n_row);
    //Q.memcpy(b_shared,b,sizeof(double)*n_row*n_row);

    for (size_t i = 0; i < n_row; i++)
    {
        b_shared[i] = b[i];
        for (size_t j = 0; j < n_row; j++)
        {
            A_shared[i*n_row+j] = A[i*n_row+j];
        }
        
    }
    

    Q.wait();


    
    auto N = static_cast<size_t>(n_row);
    sycl::range<1> global1{N};
    sycl::range<2> global2{N,N};


    auto N_b = static_cast<size_t>(block_size);
    if (block_size > n_row)
    {
        std::cout << "Given input block size is greater than the matrix size changing block size to matrix size \n" << std::endl;
        N_b = N;
    }

    sycl::range<1> local1{N_b};
    sycl::range<2> local2{N_b,N_b};   

    auto kernel_duration1 =  std::chrono::microseconds(0) ;
    auto kernel_duration2 =  std::chrono::microseconds(0) ;
    auto kernel_duration3 =  std::chrono::microseconds(0) ;
    auto kernel_duration4 =  std::chrono::microseconds(0) ;
    auto kernel_duration5 =  std::chrono::microseconds(0) ;
    auto kernel_duration6 =  std::chrono::microseconds(0) ;
    auto kernel_duration7 =  std::chrono::microseconds(0) ;


    auto kernel_start_time = std::chrono::high_resolution_clock::now();


    auto kernel_start1 = std::chrono::high_resolution_clock::now();

    Q.parallel_for<class stream>(sycl::range<1>(global1), [=](sycl::item<1>it){
        
        auto i = it.get_id(0);

        auto temp = 0.0;

        for (size_t j = 0; j < N; j++)
        {
          temp += A_shared[i*N+j]*x0[j];
        }
        
        r[i] = b_shared[i] - temp ;
        
    });

    Q.wait();

    auto kernel_end1 = std::chrono::high_resolution_clock::now();

    kernel_duration1 += std::chrono::duration_cast<std::chrono::microseconds>(kernel_end1 - kernel_start1);

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

    auto accum = sycl::malloc_shared<double>(n_row*sizeof(double),Q);


    auto tolerance = 1E-5 ;

    

    while(err > tolerance)
    {

        std::fill(accum,accum+n_row,0.0);
        num[0] = 0.0;
        den[0] = 0.0;

        //##########

        auto kernel_start2 = std::chrono::high_resolution_clock::now();

        if(atomics)
        {
            Q.parallel_for<>(sycl::range<1>(global1), [=](sycl::item<1>it){
            
                auto i = it.get_id(0);

                auto v = sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                num[0]);

                v.fetch_add(r[i]*r[i]);   
            
            });

        }
        else
        {
            auto accum_shared = sycl::malloc_shared<double>(n_row/block_size,Q); Q.wait();
            auto tile = static_cast<size_t>(block_size);
            Q.parallel_for<>(sycl::range<1>(global1), [=](sycl::item<1>it){

                auto j = it.get_id(0);

                for (size_t k = 0; k < tile; k++)
                {
                    accum_shared[j] += r[j*tile + k]*r[j*tile + k];
                }
            
            
            });

            std::accumulate(accum_shared, accum_shared+(n_row/block_size), num[0]);

        }


        auto kernel_end2 = std::chrono::high_resolution_clock::now();

        kernel_duration2 += std::chrono::duration_cast<std::chrono::microseconds>(kernel_end2 - kernel_start2); 

        //##########

        auto kernel_start3 = std::chrono::high_resolution_clock::now();

        Q.parallel_for<>(sycl::range<1>(global1), [=](sycl::item<1>it){
            
            auto i = it.get_id(0);

            for (size_t j = 0; j < N; j++)
            {
              accum[i] += p[i]*A_shared[i*N+j]*p[j] ;
            }      
            
            
        }); 


        Q.wait();

        auto kernel_end3 = std::chrono::high_resolution_clock::now();

        kernel_duration3 += std::chrono::duration_cast<std::chrono::microseconds>(kernel_end3 - kernel_start3); 

        //##########

        
        den[0] = std::accumulate(accum, accum+n_row,0.0);
                
        alpha[0] = num[0] / den[0]; 

        //##########

        auto kernel_start4 = std::chrono::high_resolution_clock::now();

        Q.parallel_for<>(sycl::range<1>(global1), [=](sycl::item<1>it){
            
            auto i = it.get_id(0);

            x0[i] = alpha[0]*p[i];      
            
        }); 

        Q.wait();

        auto kernel_end4 = std::chrono::high_resolution_clock::now();

        kernel_duration4 += std::chrono::duration_cast<std::chrono::microseconds>(kernel_end4 - kernel_start4);

        //##########

        auto kernel_start5 = std::chrono::high_resolution_clock::now();

        Q.parallel_for<>(sycl::range<1>(global1), [=](sycl::item<1>it){
            
            auto i = it.get_id(0);

            double temp = 0.0;

            for (size_t j = 0; j < N; j++)
            {
              temp+= alpha[0]*A_shared[i*N+j]*p[j];
            }    

            r[i] = r[i] - temp;
            
        });

        Q.wait();

        auto kernel_end5 = std::chrono::high_resolution_clock::now();

        kernel_duration5 += std::chrono::duration_cast<std::chrono::microseconds>(kernel_end5 - kernel_start5);

        //std::cout << kernel_duration4.count()/(k*iterations*1E6) << std::endl;

        //##########

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

        //##########

        auto kernel_start6 = std::chrono::high_resolution_clock::now();if(atomics)
        {
            Q.parallel_for<>(sycl::range<1>(global1), [=](sycl::item<1>it){
            
                auto i = it.get_id(0);

                auto v = sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                num[0]);

                v.fetch_add(r[i]*r[i]); 

                auto v1 = sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                den[0]);

                v1.fetch_add(rp[i]*rp[i]);   
            
            });

        }
        else
        {
            auto accum_shared_num = sycl::malloc_shared<double>(n_row/block_size,Q); Q.wait();
            auto accum_shared_den = sycl::malloc_shared<double>(n_row/block_size,Q); Q.wait();
            auto tile = static_cast<size_t>(block_size);
            Q.parallel_for<>(sycl::range<1>(global1), [=](sycl::item<1>it){

                auto j = it.get_id(0);

                for (size_t k = 0; k < tile; k++)
                {
                    accum_shared_num[j] += r[j*tile+k]*r[j*tile+k];
                    accum_shared_den[j] += rp[j*tile+k]*rp[j*tile+k];
                }
            
            
            });

            std::accumulate(accum_shared_num, accum_shared_num+(n_row/block_size), num[0]);
            std::accumulate(accum_shared_den, accum_shared_den+(n_row/block_size), den[0]);

        }


        auto kernel_end6 = std::chrono::high_resolution_clock::now();

        kernel_duration6 += std::chrono::duration_cast<std::chrono::microseconds>(kernel_end6 - kernel_start6);

        //##########

        beta[0] = num[0]/den[0];

        //##########

        auto kernel_start7 = std::chrono::high_resolution_clock::now();

        Q.parallel_for<>(sycl::range<1>(global1), [=](sycl::item<1>it){
            
            auto i = it.get_id(0);
    
            p[i] = r[i] + beta[0]*p[i];        
            
        });

        Q.wait();

        auto kernel_end7 = std::chrono::high_resolution_clock::now();

        kernel_duration7 += std::chrono::duration_cast<std::chrono::microseconds>(kernel_end7 - kernel_start7);

        //##########

        for (size_t i = 0; i < n_row; i++)
        {
          rp[i] = r[i];
        }

        k++;

    }

    auto kernel_end_time = std::chrono::high_resolution_clock::now();

    auto kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end_time - kernel_start_time);

    auto appl_duration = std::chrono::microseconds(0);

    appl_duration = kernel_duration2 + kernel_duration3 + kernel_duration4 + kernel_duration5 + kernel_duration6 + kernel_duration7;
 
    std::cout << "Average total time taken to execute application : "<< (kernel_duration.count()/(1E6)) <<" seconds" <<std::endl;
    std::cout << "\n";
 
    std::cout << "Average time taken to execute kernel1 : "<< kernel_duration1.count()/(1E6) <<" seconds" <<std::endl;
    std::cout << "\n"; 

    std::cout << "Average time taken to execute kernel2 : "<< kernel_duration2.count()/(1E6) <<" seconds" <<std::endl;
    std::cout << "\n"; 

    std::cout << "Average time taken to execute kernel3 : "<< kernel_duration3.count()/(1E6) <<" seconds" <<std::endl;
    std::cout << "\n"; 

    std::cout << "Average time taken to execute kernel4 : "<< kernel_duration4.count()/(1E6) <<" seconds" <<std::endl;
    std::cout << "\n"; 

    std::cout << "Average time taken to execute kernel5 : "<< kernel_duration5.count()/(1E6) <<" seconds" <<std::endl;
    std::cout << "\n"; 

    std::cout << "Average time taken to execute kernel6 : "<< kernel_duration6.count()/(1E6) <<" seconds" <<std::endl;
    std::cout << "\n";       

    std::cout << "Average time taken to execute kernel7 : "<< kernel_duration7.count()/(iterations*1E6) <<" seconds" <<std::endl;
    std::cout << "\n"; 
   
    }
  
    return 0;
    
}
