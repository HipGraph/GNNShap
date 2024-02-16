// This is GPU sampler of GNNShap.
// It is compiled as a shared library during the first run and called by the main GNNShap code.


#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <math.h>
#include <chrono>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>


#include <pybind11/pybind11.h>
#include <Python.h>



namespace py = pybind11;


// ######################DEVICE FUNCTIONS#########################


/* combination generation algorithm on cuda from
https://forums.developer.nvidia.com/t/enumerating-combinations/19980/4
*/

//calculate binomial coefficient
__inline__ __host__ __device__ unsigned int BinCoef(int n, int r) {
    unsigned int b = 1;  
    for(int i=0;i<=(r-1);i++) {
        b= (b * (n-i))/(i+1);
    }
    return(b);

    //the following is slower on CPU. I didn't test on GPU.
    //lround(std::exp( std::lgamma(n+1)-std::lgamma(n-k+1)-std::lgamma(k+1)));
}

//assigns the rth combination of n choose k to the array maskMat's row
__device__ int rthComb(unsigned int r, bool* rowPtr, bool* symRowPtr, int n, int k) { 
    int x = 1;  
    unsigned int y;
    for(int i=1; i <= k; i++) {
        y = BinCoef(n-x,k-i);
        while (y <= r) {
            r = r - y;
            x = x+1;
	        if (x > n)
		        return 0;
            y= BinCoef(n-x,k-i);
        }
        rowPtr[x-1] = true;
        symRowPtr[x-1] = false;
        x = x + 1;
    }
    return 1;

}

__global__ void cudaSampleGenerator(int nPlayers, int nHalfSamp, int* sizeLookup,
                  bool* maskMat, int rndStartInd,
                  int* devStartInds, int* devShuffleArr) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nTotalThreads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
  int chunk = nHalfSamp/nTotalThreads + 1;

  if (tid*chunk >= nHalfSamp)
    return;

  int* localShuffleArr = devShuffleArr + tid * nPlayers;
  



  bool* mask = maskMat + chunk * tid * nPlayers;
  // symmetrics starts from the middle
  bool* maskSym = maskMat + nHalfSamp * nPlayers + chunk * tid * nPlayers; 
  int i, k;

  // nchoosek based sampling
  int fullCoalTaskEndInd = min(chunk*(tid + 1), rndStartInd);
  int rndTaskEndInd = min(chunk*(tid + 1), nHalfSamp);

  for (i = tid * chunk; i < fullCoalTaskEndInd; i++) {
    k  =  sizeLookup[i];
    rthComb( i - devStartInds[k-1], mask, maskSym, nPlayers, k); //generate combination
    mask += nPlayers; //move pointer to the next combination
    maskSym += nPlayers;
  }


  if (rndTaskEndInd <= fullCoalTaskEndInd)
    return;

  
  // random sampling
  // do random sampling here!

    curandState_t state;
  curand_init(1234, tid, 0, &state);
  int temp, y, z;
  for (z = 0; z < nPlayers; z++)
    localShuffleArr[z] = z;

  for (;i < rndTaskEndInd; i++) {
      //knuthShuffle Algorithm
      for (z = nPlayers - 1; z > 0; z--) {
            y = (int)(curand_uniform(&state)*(z + .999999));
            temp = localShuffleArr[z];
            localShuffleArr[z] = localShuffleArr[y];
            localShuffleArr[y] = temp;
      }
    
    for (int j = 0; j < sizeLookup[i]; j++){
      mask[localShuffleArr[j]] = true;
      maskSym[localShuffleArr[j]] = false;
    }

    mask += nPlayers; //move pointer to the next combination
    maskSym += nPlayers;
  }
}

// ######################HOST FUNCTIONS#########################

__inline__ double arraySum(double* arr, int n) {
  double sum = 0;
  for (int i = 0; i < n; i++)
    sum += arr[i];
  return sum;
}

__inline__ int arraySum(int* arr, int n) {
  int sum = 0;
  for (int i = 0; i < n; i++)
    sum += arr[i];
  return sum;
}

__inline__ void normalizeArray(double* arr, int n){
  double sum = arraySum(arr, n);
  for (int i = 0; i < n; i++)
    arr[i] /= sum;
}

__inline__ void divideArray(double* arr, int n, double divisor){
  for (int i = 0; i < n; i++)
    arr[i] /= divisor;
}

void cudaSample(torch:: Tensor maskMatTensor, torch::Tensor kWTensor, int nPlayers,
                    int nSamples, int nBlocks = 1, int nThreads = 6) {

  int nHalfSamp = nSamples/2 ; // we will get symmetric samples. no need to compute the other half
  int nSamplesLeft = nHalfSamp; 

  int* sizeLookup = new int[nSamplesLeft];
  double* kernelWeights = new double[nSamples];

  int* tmpSLookPtr = sizeLookup;
  double* tmpKWPointer = kernelWeights;

  
  int nSubsetSizes = ceil((nPlayers - 1) / 2.0); // number of subset sizes
  // coalition size in the middle not a paired subset
  // if nPlayers=4, 1 and 3 are pairs, 2 doesn't have a pair
  int nPairedSubsetSizes = floor((nPlayers - 1) / 2.0);

  // number of samples for each subset size
  int* coalSizeNSamples = new int[nSubsetSizes];

  int* startInds = new int[nSubsetSizes+1]; // coalition size sample start indices

  // weight vector to distribute samples
  double* weightVect = new double[nSubsetSizes];


  // compute weight vector
  for (int i = 1; i <= nSubsetSizes; i++) {
    weightVect[i-1] = ((nPlayers - 1.0) / (i * (nPlayers - i)));
  }


  // we will get the symmetric except in the middle
  if (nSubsetSizes != nPairedSubsetSizes)
      weightVect[nPairedSubsetSizes] /= 2;

  // normalize weight vector to sum to 1
  normalizeArray(weightVect, nSubsetSizes);

  double * remWeightVect = new double[nSubsetSizes];
  std::copy(weightVect, weightVect + nSubsetSizes, remWeightVect);

  // std::cout << "initial remWeightVect: ";
  // for (int b = 0; b < nSubsetSizes; b++){
  //   std::cout << remWeightVect[b] << " ";
  // }
  // std::cout << std::endl;

  double sumKW = 0;
  startInds[0] = 0;

  // check if we have enough samples to iterate all coalitions for each subset size.
  int nFullSubsets = 0;
  long nSubsets;
  for(int i = 1; i <= nSubsetSizes; i++){
    nSubsets =   BinCoef(nPlayers, i);//nChoosek(nPlayers, i);

    if (i > nPairedSubsetSizes){
      if (nSubsets % 2 != 0)
        std::cout << "Error: nSubsets is not even. Be careful!!!!" << std::endl;
      nSubsets /= 2;
      // std::cout << "inside if middle full sample control case" << std::endl;
      }

    if (nSamplesLeft * remWeightVect[i-1] + 1e-8 >= nSubsets){
      nFullSubsets++;
      coalSizeNSamples[i-1] = nSubsets;
      nSamplesLeft -= nSubsets;
      startInds[i] = startInds[i-1] + nSubsets;

      sumKW += (50*weightVect[i-1]);
      std::fill(tmpKWPointer, tmpKWPointer + nSubsets, (50*weightVect[i-1]) / nSubsets);
      std::fill(tmpSLookPtr, tmpSLookPtr + nSubsets, i);

      tmpKWPointer += nSubsets;
      tmpSLookPtr += nSubsets;
      
      if (remWeightVect[i-1] < 1.0){
        divideArray(remWeightVect + i-1, nSubsetSizes -i+1, 1-remWeightVect[i-1]);
      }

    }
    else{
      break;
    }    
  }

  // use this if we want equal weights for each randomly sampled coalitions.
  double remKw = (50.0 - sumKW)/nSamplesLeft;
  std::fill(tmpKWPointer, tmpKWPointer + nSamplesLeft, remKw);
  tmpKWPointer += nSamplesLeft;

  int rndStartInd = nHalfSamp - nSamplesLeft;

  // if we have enough samples to iterate all coalitions for each subset size, then we are done.
  if (nFullSubsets != nSubsetSizes){
    int remSamples = nSamplesLeft;
    bool roundUp = true;
    for (int i = nFullSubsets; i < nSubsetSizes - 1; i++){
      
      // extra check to avoid negative number of samples for the middle coal. Might be redundant
      if (nSamplesLeft <= 0) {
        nSamplesLeft = 0;
        break;
      }

      if (roundUp)
        coalSizeNSamples[i] = min((int)ceil(remSamples * remWeightVect[i]), nSamplesLeft);
      else
        coalSizeNSamples[i] = min((int)floor(remSamples * remWeightVect[i]), nSamplesLeft);
      nSamplesLeft -= coalSizeNSamples[i];

      // if we want different weights for each randomly sampled coalition sizes, we can use this.
      // However, experiments show that it doesn't make a difference.
      //std::fill(tmpKWPointer, tmpKWPointer + coalSizeNSamples[i], (50*weightVect[i]) / coalSizeNSamples[i]);
      //tmpKWPointer += coalSizeNSamples[i];

      std::fill(tmpSLookPtr, tmpSLookPtr + coalSizeNSamples[i], i+1);
      tmpSLookPtr += coalSizeNSamples[i];

      startInds[i+1] = startInds[i] + coalSizeNSamples[i];

      roundUp = !roundUp;
    }
    //add the remaining samples to the middle coal. I removed the middle coal from the loop above
    // to avoid negative number of samples for the middle coal.
    coalSizeNSamples[nSubsetSizes-1] = nSamplesLeft;

    //startInds[nSubsetSizes-1] = startInds[nSubsetSizes-2] + nSamplesLeft;


    // uncomment this if we want different weights for each randomly sampled coalition sizes.
    // However, experiments show that it doesn't make a difference.
    // std::fill(tmpKWPointer, tmpKWPointer + nSamplesLeft, (50*remWeightVect[nSubsetSizes-1]) / nSamplesLeft);
    //tmpKWPointer += nSamplesLeft;    
    
    std::fill(tmpSLookPtr, tmpSLookPtr + nSamplesLeft, nSubsetSizes);

    if (coalSizeNSamples[nSubsetSizes-1] < 0)
      std::cout << "Error: negative number of samples for the middle coalition" << std::endl;

    nSamplesLeft = 0;
  }

  // symmetric weights. No need to compute the other half, no need to flip
  memcpy(tmpKWPointer, kernelWeights, nHalfSamp * sizeof(double));

  bool *devMaskMat = maskMatTensor.data_ptr<bool>();
  // cudaMalloc(&devMaskMat, nSamples * nPlayers * sizeof(bool));
  // cudaMemset(devMaskMat, false, nHalfSamp * nPlayers * sizeof(bool));
  cudaMemset(devMaskMat + nPlayers * nHalfSamp, true, nHalfSamp * nPlayers * sizeof(bool));

  int *deviceSizeLookup;
  cudaMalloc(&deviceSizeLookup, nHalfSamp * sizeof(int));
  cudaMemcpy(deviceSizeLookup, sizeLookup, nHalfSamp * sizeof(int), cudaMemcpyHostToDevice);


  int* devShuffleArr;
  cudaMalloc(&devShuffleArr, nBlocks * nThreads * nPlayers * sizeof(int));

  int* devStartInds; // device start indices
  cudaMalloc(&devStartInds, nSubsetSizes * sizeof(int));
  cudaMemcpy(devStartInds, startInds, nSubsetSizes * sizeof(int), cudaMemcpyHostToDevice);

  cudaSampleGenerator<<<nBlocks, nThreads>>>(nPlayers, nHalfSamp, deviceSizeLookup, 
                                            devMaskMat, rndStartInd, devStartInds, devShuffleArr);

  
  cudaMemcpy(kWTensor.data_ptr<double>(), kernelWeights,
            nSamples * sizeof(double), cudaMemcpyHostToDevice);
  
  cudaDeviceSynchronize();

  free(sizeLookup);
  free(kernelWeights);
  free(coalSizeNSamples);
  free(startInds);
  free(weightVect);
  free(remWeightVect);
  cudaFree(deviceSizeLookup);
  cudaFree(devStartInds);
  cudaFree(devShuffleArr);
}



// ######################PYTHON BINDINGS#########################




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample", &cudaSample, "Cuda Sample");
}