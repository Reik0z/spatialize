#ifndef _SPTLZ_AGG_
#define _SPTLZ_AGG_

#include <cmath>

namespace sptlz{
  float average(std::vector<float> v){
    int n = 0;
    float c = 0.0;

    for(float f:v){
      if(!std::isnan(f)){
        c += f;
        n++;
      }
    }
    return(c/n);
  }

  std::vector<float> average(std::vector<std::vector<float>> m){
    std::vector<float> result;
    for(std::vector<float> v:m){
      result.push_back(average(v));
    }
    return(result);
  }

}

#endif
