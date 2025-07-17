#ifndef _SPTLZ_ADTV_ESI_IDW_
#define _SPTLZ_ADTV_ESI_IDW_

#include <stdexcept>
#include <cmath>
#include <random>
#include "spatialize/abstract_esi.hpp"
#include "spatialize/utils.hpp"

namespace sptlz{
  class LOO2D{
    protected:
      std::vector<std::vector<float>> *coords;
      std::vector<float> *coords1d;
      std::vector<float> *values;
      std::vector<float> centroid;

    public:
      LOO2D(std::vector<std::vector<float>> *_coords, std::vector<float> *_values){
        values = _values;
        centroid = sptlz::get_centroid(_coords);
        coords = _coords;
        auto aux = sptlz::as_1d_array(_coords);
        coords1d = &aux;
      }

      float eval(std::vector<float> X){
        int n = values->size();
        float r = 0.0, sum_w, est, wj;
        std::vector<float> params = {X.at(1), X.at(2)};
        auto tr_coords = sptlz::transform(coords, &params, &centroid);

        for(int i=0; i<n; i++){
          auto ds = sptlz::distances(&tr_coords, i);
          sum_w = 0.0;
          est = 0.0;
          for(int j=0;j<n;j++){
            if(j!=i){
              wj = 1.0/(1.0+std::pow(ds.at(j), X.at(0)));
              sum_w += wj;
              est += wj*values->at(j);
            }
          }
          r += std::pow(values->at(i)-est/sum_w, 2.0);
        }
        return(r/n);
      }
  };

  class LOO3D{
    protected:
      std::vector<std::vector<float>> *coords;
      std::vector<float> *coords1d;
      std::vector<float> *values;
      std::vector<float> centroid;

    public:
      LOO3D(std::vector<std::vector<float>> *_coords, std::vector<float> *_values){
        values = _values;
        centroid = sptlz::get_centroid(_coords);
        coords = _coords;
        auto aux = sptlz::as_1d_array(_coords);
        coords1d = &aux;
      }

      float eval(std::vector<float> X){
        int n = values->size();
        float r = 0.0, sum_w, est, wj;
        std::vector<float> params = {X.at(1), X.at(2), X.at(3), X.at(4), X.at(5)};
        auto tr_coords = sptlz::transform(coords, &params, &centroid);

        for(int i=0; i<n; i++){
          auto ds = sptlz::distances(&tr_coords, i);
          sum_w = 0.0;
          est = 0.0;
          for(int j=0;j<n;j++){
            if(j!=i){
              wj = 1.0/(1.0+std::pow(ds.at(j), X.at(0)));
              sum_w += wj;
              est += wj*values->at(j);
            }
          }
          r += std::pow(values->at(i)-est/sum_w, 2.0);
        }
        return(r/n);
      }
  };

  class ADAPTIVE_ESI_IDW: public ESI {
    protected:
      std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params){
        std::vector<float> result;

        if(locations_id->size()==0){
          return(result);
        }

        if(samples_id->size()==0){
          for([[maybe_unused]] auto l: *locations_id){
            result.push_back(NAN);
          }
          return(result);
        }

        if(samples_id->size()==1){
          for([[maybe_unused]] auto l: *locations_id){
            result.push_back(params->at(0));
          }
          return(result);
        }

        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        auto sl_locations = slice(locations, locations_id);
        std::vector<float> centroid;
        int i_params = 0;

        centroid.push_back(params->at(i_params++));
        centroid.push_back(params->at(i_params++));

        if(coords->at(0).size()==3){
          centroid.push_back(params->at(i_params++));
        }

        float exponent = params->at(i_params++);
        std::vector<float> rot_params = slice_from(params, i_params);

        auto tr_coords = transform(&sl_coords, &rot_params, &centroid);
        auto tr_locations = transform(&sl_locations, &rot_params, &centroid);

        float w, w_sum, w_v_sum;

        // for every location
        for(int i=0; i<locations_id->size(); i++){
          w_sum = 0.0;
          w_v_sum = 0.0;

          for(int j=0; j<samples_id->size(); j++){
            // calculate weight
            w = 1/(1+std::pow(distance(&(tr_locations.at(i)), &(tr_coords.at(j))), exponent));
            // keep sum of weighted values and sum of weights
            w_sum += w;
            w_v_sum += w*sl_values.at(j);
          }
          // return weighted values sum normalized (divided by weights sum)
          result.push_back(w_v_sum/w_sum);
        }

        return(result);
      }

      std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params){
        std::vector<float> result;

        if((samples_id->size()==0) || (samples_id->size()==1)){
          for([[maybe_unused]] auto l: *samples_id){
            result.push_back(NAN);
          }
          return(result);
        }

        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        std::vector<float> centroid;
        int i_params = 0;
        centroid.push_back(params->at(i_params++));
        centroid.push_back(params->at(i_params++));
        if(coords->at(0).size()==3){
          centroid.push_back(params->at(i_params++));
        }

        float exponent = params->at(i_params++);
        std::vector<float> rot_params = slice_from(params, i_params);

        auto tr_coords = transform(&sl_coords, &rot_params, &centroid);

        float w, w_sum, w_v_sum;

        // for every location
        for(int i=0; i<samples_id->size(); i++){
          w_sum = 0.0;
          w_v_sum = 0.0;

          for(int j=0; j<samples_id->size(); j++){
            if(i!=j){
              // calculate weight
              w = 1/(1+std::pow(distance(&(tr_coords.at(i)), &(tr_coords.at(j))), exponent));
              // keep sum of weighted values and sum of weights
              w_sum += w;
              w_v_sum += w*sl_values.at(j);
            }
          }
          // return weighted values sum normalized (divided by weights sum)
          result.push_back(w_v_sum/w_sum);
        }
        return(result);
      }

      std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *folds, std::vector<int> *samples_id, std::vector<float> *params){
        std::vector<float> result(samples_id->size());
        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        auto sl_folds = slice(folds, samples_id);
        float w, w_sum, w_v_sum;

        if((samples_id->size()==0) || (samples_id->size()==1)){
          for([[maybe_unused]] auto l: *samples_id){
            result.push_back(NAN);
          }
          return(result);
        }

        std::vector<float> centroid;
        int i_params = 0;
        centroid.push_back(params->at(i_params++));
        centroid.push_back(params->at(i_params++));
        if(coords->at(0).size()==3){
          centroid.push_back(params->at(i_params++));
        }

        float exponent = params->at(i_params++);
        std::vector<float> rot_params = slice_from(params, i_params);
        auto tr_coords = transform(&sl_coords, &rot_params, &centroid);

        for(int i=0; i<k; i++){
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
                  w = 1/(1+std::pow(distance(&(tr_coords.at(j)), &(tr_coords.at(l))), exponent));
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

      void post_process(){
        std::vector<std::vector<float>> leaf_coords;
        std::vector<float> leaf_values;
        sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
        sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

        logger->info("computing optimal parameters");

        progress->init(mondrian_forest.size(), 1);

        for(int i=0; i<mondrian_forest.size(); i++){
          auto mt = mondrian_forest.at(i);
          for(int j=0; j<mt->samples_by_leaf.size(); j++){
            leaf_coords.clear();
            leaf_values.clear();
            for(int k=0; k<mt->samples_by_leaf.at(j).size(); k++){
              leaf_coords.push_back(coords.at(mt->samples_by_leaf.at(j).at(k)));
              leaf_values.push_back(values.at(mt->samples_by_leaf.at(j).at(k)));
            }

            mt->leaf_params.at(j) = get_params(&leaf_coords, &leaf_values);
          }

          if (PyErr_CheckSignals() != 0)  // to allow ctrl-c from user
             exit(0);
		  progress->inform(i + 1);
        }

        progress->stop();

        delete logger;
        delete progress;
      }

      std::vector<float> get_params(std::vector<std::vector<float>> *coords, std::vector<float> *values){
        std::uniform_real_distribution<float> uni_float(0, 1);
        int best_of = 3;
        if(coords->size()==0){
          return(std::vector<float>());
        }else if(coords->size()==1){
          return(std::vector<float>({values->at(0)}));
        }

        std::vector<float> min_coords;
        if(coords->at(0).size()==2){
          std::vector<float> starting_point, candidate;
          float min_value=1e20, aux;
          LOO2D *func = new LOO2D(coords, values);
          std::vector<std::vector<float>> ranges = {
            {0.1, 8.0, 0.2}, // exp
            {0.0, 180.0, 1.0}, // azim
            {0.1, 1.001, 0.05} // ratio
          };
          for(int i=0; i<best_of; i++){
            starting_point = {};
            for(int j=0; j<ranges.size(); j++){
              starting_point.push_back(ranges.at(j).at(0)+uni_float(my_rand)*(ranges.at(j).at(1)-ranges.at(j).at(0)));
            }
            candidate = sptlz::grid_search<LOO2D>(func, &ranges, starting_point);
            aux = func->eval(candidate);
            if(aux<min_value){
              min_coords = candidate;
              min_value = aux;
            }
          }
        }else if(coords->at(0).size()==3){
          std::vector<float> starting_point, candidate;
          float min_value=1e20, aux;
          LOO3D *func = new LOO3D(coords, values);
          std::vector<std::vector<float>> ranges = {
            {0.1, 8.0, 0.2}, // exp
            {0.0, 180.0, 1.0}, // azim
            {0.0, 180.0, 1.0}, // dip
            {0.0, 180.0, 1.0}, // plunge
            {0.1, 1.001, 0.05}, // ratio1
            {0.1, 1.001, 0.05} // ratio2
          };
          for(int i=0; i<best_of; i++){
            starting_point = {};
            for(int j=0; j<ranges.size(); j++){
              starting_point.push_back(ranges.at(j).at(0)+uni_float(my_rand)*(ranges.at(j).at(1)-ranges.at(j).at(0)));
            }
            candidate = sptlz::grid_search<LOO3D>(func, &ranges, starting_point);
            aux = func->eval(candidate);
            if(aux<min_value){
              min_coords = candidate;
              min_value = aux;
            }
          }
        }

        if(min_coords.size()==0){ // don't know why sometimes it can't get a candidate
          if (coords->at(0).size()==2){
            min_coords = {2.0, 0.0, 1.0};
          }else if(coords->at(0).size()==3){
            min_coords = {2.0, 0.0, 0.0, 0.0, 1.0, 1.0};
          }
        }
        auto centroid = sptlz::get_centroid(coords);
        for(auto v: min_coords){
          centroid.push_back(v);
        }

        return(centroid);
      }

    public:
      ADAPTIVE_ESI_IDW(std::vector<std::vector<float>> _coords,
                       std::vector<float> _values,
                       float lambda,
                       int forest_size,
                       std::vector<std::vector<float>> bbox,
                       std::function<int(std::string)> visitor,
                       int seed=206936):
      ESI(_coords, _values, lambda, forest_size, bbox, visitor, seed){
        this->class_name = __func__;
        post_process();
      }

      ADAPTIVE_ESI_IDW(std::vector<sptlz::MondrianTree*> _mondrian_forest,
                       std::vector<std::vector<float>> _coords,
                       std::vector<float> _values,
                       std::function<int(std::string)> visitor):
      ESI(_mondrian_forest, _coords, _values, visitor){
        this->class_name = __func__;
      }
  };
}

#endif
