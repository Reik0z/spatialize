#ifndef _SPTLZ_CUSTOM_ESI_
#define _SPTLZ_CUSTOM_ESI_

#include <stdexcept>
#include <cmath>
#include <functional>
#include "spatialize/abstract_esi.hpp"
#include "spatialize/utils.hpp"

namespace sptlz{
  class CUSTOM_ESI: public ESI {
    protected:
      std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*)> post_creation;
      std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*, std::vector<std::vector<float>>*, std::vector<float> *)> estimation_by_leaf;
      std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*, std::vector<float> *)> loo_by_leaf;
      std::function<std::vector<float>(int, std::vector<std::vector<float>>*, std::vector<float>*, std::vector<int> *, std::vector<float> *)> kfold_by_leaf;

      std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params){
        auto _coords = slice(coords, samples_id);
        auto _values = slice(values, samples_id);
        auto _locations = slice(locations, locations_id);
        auto result = estimation_by_leaf(&_coords, &_values, &_locations, params);
        return(result);
      }

      std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params){
        auto _coords = slice(coords, samples_id);
        auto _values = slice(values, samples_id);
        auto result = loo_by_leaf(&_coords, &_values, params);
        return(result);
      }

      std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *folds, std::vector<int> *samples_id, std::vector<float> *params){
        auto _coords = slice(coords, samples_id);
        auto _values = slice(values, samples_id);
        auto _folds = slice(folds, samples_id);
        auto result = kfold_by_leaf(k, &_coords, &_values, &_folds, params);
        return(result);
      }

      void post_process(){
        if(post_creation == NULL){
            return;
        }
        
        std::vector<std::vector<float>> leaf_coords;
        std::vector<float> leaf_values;

        for(int i=0; i<mondrian_forest.size(); i++){
          auto mt = mondrian_forest.at(i);
          for(int j=0; j<mt->samples_by_leaf.size(); j++){
            leaf_coords.clear();
            leaf_values.clear();
            for(int k=0; k<mt->samples_by_leaf.at(j).size(); k++){
              leaf_coords.push_back(coords.at(mt->samples_by_leaf.at(j).at(k)));
              leaf_values.push_back(values.at(mt->samples_by_leaf.at(j).at(k)));
            }
            mt->leaf_params.at(j) = post_creation(&leaf_coords, &leaf_values);
          }
        }
      }

    public:
      CUSTOM_ESI( std::vector<std::vector<float>> _coords, 
                  std::vector<float> _values, 
                  float lambda, 
                  int forest_size, 
                  std::vector<std::vector<float>> bbox, 
                  std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*)> _post,
                  std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*, std::vector<std::vector<float>>*, std::vector<float> *)> _est,
                  std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*, std::vector<float> *)> _loo,
                  std::function<std::vector<float>(int, std::vector<std::vector<float>>*, std::vector<float>*, std::vector<int> *, std::vector<float> *)> _kfold,
                  std::function<int(std::string)> visitor,
                  int seed=206936):
      ESI(_coords, _values, lambda, forest_size, bbox, visitor, seed){
        post_creation = _post;
        estimation_by_leaf = _est;
        loo_by_leaf = _loo;
        kfold_by_leaf = _kfold;
      }

      CUSTOM_ESI( std::vector<sptlz::MondrianTree*> _mondrian_forest, 
                  std::vector<std::vector<float>> _coords, 
                  std::vector<float> _values,
                  std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*)> _post,
                  std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*, std::vector<std::vector<float>>*, std::vector<float> *)> _est,
                  std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*, std::vector<float> *)> _loo,
                  std::function<std::vector<float>(int, std::vector<std::vector<float>>*, std::vector<float>*, std::vector<int> *, std::vector<float> *)> _kfold,
                  std::function<int(std::string)> visitor):
      ESI(_mondrian_forest, _coords, _values, visitor){
        post_creation = _post;
        estimation_by_leaf = _est;
        loo_by_leaf = _loo;
        kfold_by_leaf = _kfold;
      }

      ~CUSTOM_ESI() {}
  };
}

#endif

