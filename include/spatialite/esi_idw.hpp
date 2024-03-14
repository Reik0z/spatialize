#ifndef _SPTLZ_ESI_IDW_
#define _SPTLZ_ESI_IDW_

#include <stdexcept>
#include <cmath>
#include "abstract_esi.hpp"
#include "utils.hpp"

namespace sptlz{
  class ESI_IDW: public ESI {
    protected:
      float exponent;

      // we do not need extra info
      std::vector<float> *get_extra(int leaf_id){
        return(NULL);
      }

      std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *queries, std::vector<int> *queries_id, std::vector<float> *extra){
        std::vector<float> result;
        float w, w_sum, w_v_sum;
        int i, j, nq, ns;

        if(samples_id->size()==0){
          nq = queries_id->size();
          for(i=0; i<nq; i++){
            result.push_back(NAN);
          }
          return(result);
        }

        // for every location
        nq = queries_id->size();
        for(i=0; i<nq; i++){
          w_sum = 0.0;
          w_v_sum = 0.0;
          ns = samples_id->size();
          for(j=0; j<ns; j++){
            // calculate weight
            w = 1/(1+std::pow(distance(&(queries->at(queries_id->at(i))), &(coords->at(samples_id->at(j)))), exponent));
            // keep sum of weighted values and sum of weights
            w_sum += w;
            w_v_sum += w*values->at(samples_id->at(j));
          }
          // return weighted values sum normalized (divided by weights sum)
          result.push_back(w_v_sum/w_sum);
        }
        return(result);
      }

      std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *extra){
        std::vector<float> result;
        int i, j, ns;

        if((samples_id->size()==0) || (samples_id->size()==1)){
          ns = samples_id->size();
          for(i=0; i<ns; i++){
            result.push_back(NAN);
          }
          return(result);
        }

        float w, w_sum, w_v_sum;

        // for every location
        ns = samples_id->size();
        for(i=0; i<ns; i++){
          w_sum = 0.0;
          w_v_sum = 0.0;
          for(j=0; j<ns; j++){
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

      std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *folds, std::vector<int> *samples_id, std::vector<float> *extra){
        std::vector<float> result(samples_id->size());
        int i, ns;

        if((samples_id->size()==0) || (samples_id->size()==1)){
          ns = samples_id->size();
          for(i=0; i<ns; i++){
            result.push_back(NAN);
          }
          return(result);
        }

        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        auto sl_folds = slice(folds, samples_id);
        float w, w_sum, w_v_sum;

        for(i=0; i<k; i++){
          auto test_train = indexes_by_predicate<int>(&sl_folds, [i](int *j){return(*j==i);});
          if(test_train.first.size()!=0){ // if is 0, then there's nothing to estimate
            if(test_train.second.size()==0){
              for(int j: test_train.first){
                result.at(j) = NAN;
              }
            }else{
              for(int j: test_train.first){
                w_sum = 0.0;
                w_v_sum = 0.0;
                for(int l: test_train.second){
                  w = 1/(1+std::pow(distance(&(sl_coords.at(j)), &(sl_coords.at(l))), exponent));
                  w_sum += w;
                  w_v_sum += w*values->at(samples_id->at(l));
                }
                result.at(j) = w_v_sum/w_sum;
              }
            }
          }
        }
        return(result);
      }

    public:
      ESI_IDW(std::string path, std::vector<std::vector<float>> *_coords, std::vector<float> *_values, int n_tree, float alpha, std::vector<float> bbox, float _exponent, int seed=2007203, int random_range=206936):ESI(path, _coords, _values, n_tree, alpha, bbox, seed, random_range){
        this->exponent = _exponent;
        // store esi type
        this->set_param("esi_type", "idw");
        // store parameters
        this->set_param("exponent", this->exponent);
      }

      ESI_IDW(std::string path):ESI(path){
        // retrieve parameters
        this->exponent = this->get_float_param("exponent");
      }

      float get_exponent(){
        return(this->exponent);
      }
  };
}

#endif
