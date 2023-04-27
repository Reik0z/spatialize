#ifndef _SPTLZ_ESI_IDW_
#define _SPTLZ_ESI_IDW_

#include <stdexcept>
#include <cmath>
#include "spatialize/abstract_esi.hpp"
#include "spatialize/utils.hpp"

namespace sptlz{
  class ESI_IDW: public ESI {
    protected:
      float exponent;

      std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params){
        std::vector<float> result;
        float w, w_sum, w_v_sum;

        // for every location
        for(int i=0; i<locations_id->size(); i++){
          w_sum = 0.0;
          w_v_sum = 0.0;

          for(int j=0; j<samples_id->size(); j++){
            // calculate weight
            w = 1/(1+std::pow(distance(&(locations->at(locations_id->at(i))), &(coords->at(samples_id->at(j)))), exponent));
            // keep sum of weighted values and sum of weights
            w_sum += w;
            w_v_sum += w*values->at(samples_id->at(j));
          }
          // return weighted values sum normalized (divided by weights sum)
          result.push_back(w_v_sum/w_sum);
        }
        return(result);
      }

      std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params){
        std::vector<float> result;
        if(samples_id->size()==1){
          result.push_back(NAN);
          return(result);
        }
        float w, w_sum, w_v_sum;

        // for every location
        for(int i=0; i<samples_id->size(); i++){
          w_sum = 0.0;
          w_v_sum = 0.0;

          for(int j=0; j<samples_id->size(); j++){
            if(i!=j){
              // calculate weight
              w = 1/(1+std::pow(distance(&(coords->at(samples_id->at(i))), &(coords->at(samples_id->at(j)))), exponent));
              // keep sum of weighted values and sum of weights
              w_sum += w;
              w_v_sum += w*values->at(samples_id->at(j));}
          }
          // return weighted values sum normalized (divided by weights sum)
          result.push_back(w_v_sum/w_sum);
        }
        return(result);
      }

    public:
      ESI_IDW(std::vector<std::vector<float>> _coords, std::vector<float> _values, float lambda, int forest_size, std::vector<std::vector<float>> bbox, float _exponent, float seed=0):ESI(_coords, _values, lambda, forest_size, bbox, seed){
        exponent = _exponent;
      }
  };
}

#endif
