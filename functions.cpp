#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cstdlib>
#include <random>
#include <fstream>
#include <algorithm> 
#include <iostream>

#include "utils.hpp"

template<typename dtype>
func_ret_t create_matrix(dtype ** mp, int size){
  dtype * m;
  int i,j;
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<dtype> dist_uniform(0,1);
  
  dtype lamda = -0.001;
  dtype coe[2*size-1];
  dtype coe_i =0.0;


  for (i=0; i < size; i++)
  {
    coe_i = 10*exp(lamda*i); 
    j=size-1+i;     
    coe[j]=coe_i;
    j=size-1-i;     
    coe[j]=coe_i;
  }

  time_t t;

  srand((unsigned) time(&t));

  m = (dtype*) malloc(sizeof(dtype)*size*size);
  if ( m == NULL) {
    return RET_FAILURE;
  }

  for (i=0; i < size; i++) {
    for (j=i; j < size; j++) {
        auto temp = (double)(rand()%10);
        m[i*size+j] = temp;
        m[j*size+i] = temp;
//      m[i*size+j]=coe[size-1-i+j];
//      m[j*size+i]=coe[size-1-i+j];
//      m[j*size+i]=coe[size-1-i+j];
//      dtype ran = dist_uniform(rng);
//      m[i*size+j] = m[j*size+i] =ran;
    }
  }

  *mp = m;

  return RET_SUCCESS;
}

 
  

// create dense vector 
template<typename dtype>
func_ret_t create_vector(dtype **vp, int size){
  dtype *m;
  int i,j;
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<dtype> dist_uniform(0,1);

  dtype lamda = -0.001;
  dtype coe[2*size-1];
  dtype coe_i =0.0;

  for (i=0; i < size; i++)
  {
    coe_i = 10*exp(lamda*i); 
    j=size-1+i;     
    coe[j]=coe_i;
    j=size-1-i;     
    coe[j]=coe_i;
  }

  time_t t;

  srand((unsigned) time(&t));
  
  m = (dtype*) malloc(sizeof(dtype)*size);
  if ( m == NULL) {
    return RET_FAILURE;
  }

  for (i=0; i < size; i++) {
    m[i]=(double)(rand()%10);
//    m[i]=coe[size-1-i];
//    dtype ran = dist_uniform(rng);
//    m[i] = ran;

  }

  *vp = m;


  return RET_SUCCESS;
}