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
    double *  b, * x0;
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
        std::cout<<"Usage: "<< argv[0]<< "[-s matrix_size|-b blocksize <optional>| -a yes or no (for atomics) <optional>] \n" << std::endl;
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

    double* r = (double*) malloc(sizeof(double)*n_row);
    double* rp = (double*) malloc(sizeof(double)*n_row);

    double* p = (double*) malloc(sizeof(double)*n_row);

    double* alpha = (double*) malloc(sizeof(double)*1);
    double* beta = (double*) malloc(sizeof(double)*1);
    double* num = (double*) malloc(sizeof(double)*1); num[0] = 0.0;
    double* den = (double*) malloc(sizeof(double)*1); den[0] = 0.0;

    int k = 0;

    x0 = (double*)malloc(sizeof(double)*n_row);

    std::fill(x0,x0+n_row,0.0);

    { // SYCL scope

    
    sycl::queue Q{};
    std::cout << "running on ..."<< std::endl;
    std::cout << Q.get_device().get_info<sycl::info::device::name>()<<"\n"<<std::endl;

    const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

    sycl::buffer<double , 1> A_buf(A,n_row*n_col,props);
    sycl::buffer<double , 1> b_buf(b,n_row,props);
    sycl::buffer<double , 1> x0_buf(x0,n_row,props);
    sycl::buffer<double , 1> r_buf(r,n_row,props);
    sycl::buffer<double , 1> rp_buf(rp,n_row,props);
    sycl::buffer<double , 1> p_buf(p,n_row,props);
    sycl::buffer<double , 1> alpha_buf(alpha,1,props);
    sycl::buffer<double , 1> beta_buf(beta,1,props);
    sycl::buffer<double , 1> num_buf(num,1,props);
    sycl::buffer<double , 1> den_buf(den,1,props);
    
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

    Q.submit([&](sycl::handler& cgh){
    auto A_acc = A_buf.get_access<sycl::access::mode::read>(cgh);
    auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
    auto x0_acc = x0_buf.get_access<sycl::access::mode::read>(cgh);
    auto r_acc = r_buf.get_access<sycl::access::mode::read_write>(cgh);
    
    
      cgh.parallel_for<class stream>(sycl::range<1>(global1), [=](sycl::item<1>it){
          
          auto i = it.get_id(0);

          auto temp = 0.0;

          for (size_t j = 0; j < N; j++)
          {
            temp += A_acc[i*N+j]*x0_acc[j];
          }
          
          r_acc[i] = b_acc[i] - temp ;
          
      });      

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

    double* accum = (double*) malloc(sizeof(double)*n_row);

    sycl::buffer<double , 1> accum_buf(accum,n_row,props);

    auto tolerance = 1E-5 ;

    

    while(err > tolerance)
    {

        std::fill(accum,accum+n_row,0.0);
        num[0] = 0.0;
        den[0] = 0.0;

        //##########

        auto kernel_start2 = std::chrono::high_resolution_clock::now();

        if (atomics)
        {
            Q.submit([&](sycl::handler& cgh){
            auto r_acc = r_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto num_acc = num_buf.get_access<sycl::access::mode::read_write>(cgh);
            
                cgh.parallel_for<>(sycl::range<1>(global1), [=](sycl::item<1>it){
                    
                    auto i = it.get_id(0);

                    auto v = sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>(
                    num_acc[0]);

                    v.fetch_add(r_acc[i]*r_acc[i]);   
                    
                });      

            });
                
        }
        else
        {
            double* accum_atom = (double*) malloc(sizeof(double)*n_row);

            sycl::buffer<double , 1> accum_atom_buf(accum_atom,n_row,props);
            auto N = static_cast<size_t>(n_row/block_size);
            sycl::range<1> global{N};
            auto tile = static_cast<size_t>(block_size);

            Q.submit([&](sycl::handler& cgh){
            auto r_acc = r_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto accum_atom_acc = accum_atom_buf.get_access<sycl::access::mode::read_write>(cgh);
            
                cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){
                    
                    auto j = it.get_id(0);

                    for (size_t k = 0; k < tile; k++)
                    {
                        accum_atom_acc[j] += r_acc[j*tile + k]*r_acc[j*tile + k];
                    }   
                    
                });      

            });

            std::accumulate(accum_atom, accum_atom+(n_row/block_size), num[0]);

        }


        Q.wait();

        auto kernel_end2 = std::chrono::high_resolution_clock::now();

        kernel_duration2 += std::chrono::duration_cast<std::chrono::microseconds>(kernel_end2 - kernel_start2); 

        //##########

        auto kernel_start3 = std::chrono::high_resolution_clock::now();

        Q.submit([&](sycl::handler& cgh){
        auto A_acc = A_buf.get_access<sycl::access::mode::read>(cgh);
        auto p_acc = p_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto accum_acc = accum_buf.get_access<sycl::access::mode::read_write>(cgh);
        
        
          cgh.parallel_for<>(sycl::range<1>(global1), [=](sycl::item<1>it){
              
              auto i = it.get_id(0);

              for (size_t j = 0; j < N; j++)
              {
                accum_acc[i] += p_acc[i]*A_acc[i*N+j]*p_acc[j] ;
              }      
              
              
          });      

        });
        Q.wait();

        auto kernel_end3 = std::chrono::high_resolution_clock::now();

        kernel_duration3 += std::chrono::duration_cast<std::chrono::microseconds>(kernel_end3 - kernel_start3); 

        //##########

        
        den[0] = std::accumulate(accum, accum+n_row,0.0);
                
        alpha[0] = num[0] / den[0]; 

        //##########

        auto kernel_start4 = std::chrono::high_resolution_clock::now();

        Q.submit([&](sycl::handler& cgh){
        auto p_acc = p_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto alpha_acc = alpha_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto x0_acc = x0_buf.get_access<sycl::access::mode::read_write>(cgh);
        
          cgh.parallel_for<>(sycl::range<1>(global1), [=](sycl::item<1>it){
              
              auto i = it.get_id(0);

              x0_acc[i] = alpha_acc[0]*p_acc[i];      
              
          });      

        });
        Q.wait();

        auto kernel_end4 = std::chrono::high_resolution_clock::now();

        kernel_duration4 += std::chrono::duration_cast<std::chrono::microseconds>(kernel_end4 - kernel_start4);

        //##########

        auto kernel_start5 = std::chrono::high_resolution_clock::now();

        Q.submit([&](sycl::handler& cgh){
        auto A_acc = A_buf.get_access<sycl::access::mode::read>(cgh);
        auto r_acc = r_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto p_acc = p_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto alpha_acc = alpha_buf.get_access<sycl::access::mode::read_write>(cgh);
              
          cgh.parallel_for<>(sycl::range<1>(global1), [=](sycl::item<1>it){
              
              auto i = it.get_id(0);

              double temp = 0.0;

              for (size_t j = 0; j < N; j++)
              {
                temp+= alpha_acc[0]*A_acc[i*N+j]*p_acc[j];
              }    

              r_acc[i] = r_acc[i] - temp;
              
          });      

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

        auto kernel_start6 = std::chrono::high_resolution_clock::now();

        if (atomics)
        {

            Q.submit([&](sycl::handler& cgh){
            auto r_acc = r_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto rp_acc = rp_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto num_acc = num_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto den_acc = den_buf.get_access<sycl::access::mode::read_write>(cgh);

            
                cgh.parallel_for<>(sycl::nd_range<1>(global1,local1), [=](sycl::nd_item<1>it){
                    
                    auto i = it.get_global_id(0);

                    auto v = sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>(
                    num_acc[0]);

                    v.fetch_add(r_acc[i]*r_acc[i]); 

                    auto v1 = sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>(
                    den_acc[0]);

                    v1.fetch_add(rp_acc[i]*rp_acc[i]); 
                    
                });      

            });
            
        }
        else
        {

            double* accum_atom_num = (double*) malloc(sizeof(double)*n_row);
            double* accum_atom_den = (double*) malloc(sizeof(double)*n_row);

            sycl::buffer<double , 1> accum_atom_num_buf(accum_atom_num,n_row,props);
            sycl::buffer<double , 1> accum_atom_den_buf(accum_atom_den,n_row,props);

            auto N = static_cast<size_t>(n_row/block_size);
            sycl::range<1> global{N};
            auto tile = static_cast<size_t>(block_size);

            Q.submit([&](sycl::handler& cgh){
            auto r_acc = r_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto rp_acc = rp_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto accum_atom_num_acc = accum_atom_num_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto accum_atom_den_acc = accum_atom_den_buf.get_access<sycl::access::mode::read_write>(cgh);

            
                cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){
                    
                    auto j = it.get_id(0);

                    for (size_t k = 0; k < tile; k++)
                    {
                        accum_atom_num_acc[j] += r_acc[j*tile+k]*r_acc[j*tile+k];
                        accum_atom_den_acc[j] += rp_acc[j*tile+k]*rp_acc[j*tile+k];
                    }
                    
                });      

            });

            std::accumulate(accum_atom_num, accum_atom_num+(n_row/block_size), num[0]);
            std::accumulate(accum_atom_den, accum_atom_den+(n_row/block_size), den[0]);

        }



        Q.wait();

        auto kernel_end6 = std::chrono::high_resolution_clock::now();

        kernel_duration6 += std::chrono::duration_cast<std::chrono::microseconds>(kernel_end6 - kernel_start6);

        //##########

        beta[0] = num[0]/den[0];

        //##########

        auto kernel_start7 = std::chrono::high_resolution_clock::now();

        Q.submit([&](sycl::handler& cgh){
        auto r_acc = r_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto p_acc = p_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto beta_acc = beta_buf.get_access<sycl::access::mode::read_write>(cgh);
        
        
          cgh.parallel_for<>(sycl::range<1>(global1), [=](sycl::item<1>it){
              
              auto i = it.get_id(0);
      
              p_acc[i] = r_acc[i] + beta_acc[0]*p_acc[i];        
              
          });      

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
    
    std::cout << "Average total time taken to execute application : "<< (kernel_duration.count()/(iterations*1E6)) <<" seconds" <<std::endl;
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

    std::cout << "Average time taken to execute kernel7 : "<< kernel_duration7.count()/(1E6) <<" seconds" <<std::endl;
    std::cout << "\n"; 
   
    }
  
    return 0;
    
}
