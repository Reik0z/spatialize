#ifndef _SPTLZ_ESI_KRIGING_
#define _SPTLZ_ESI_KRIGING_

#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include "spatialize/abstract_esi.hpp"
#include "spatialize/utils.hpp"

namespace sptlz{
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> COV_NxN;

  std::vector<float> distances(std::vector<std::vector<float>> *coords){
    int n = coords->size();
    std::vector<float> result;

    for(int i=0; i<n; i++){
      for(int j=0; j<i; j++){
        result.push_back(result.at(j*n+i));
      }
      for(int j=i; j<n; j++){
        result.push_back(distance(&(coords->at(i)), &(coords->at(j))));
      }
    }
    return(result);
  }

  std::vector<float> kriging_right_matrix(std::vector<std::vector<float>> *coords, std::vector<std::vector<float>> *locations, std::function<float(float)> gamma){
    int n = coords->size(), m = locations->size();
    std::vector<float> result;

    for(int i=0; i<m; i++){
      for(int j=0; j<n; j++){
        result.push_back(gamma(distance(&(locations->at(i)), &(coords->at(j)))));
      }
      result.push_back(1.0);
    }
    return(result);
  }

  std::vector<float> kriging_left_matrix(std::vector<std::vector<float>> *coords, std::function<float(float)> gamma){
    int n = coords->size();
    std::vector<float> result;

    for(int i=0; i<n; i++){
      for(int j=0; j<i; j++){
        result.push_back(result.at(j*(n+1)+i));
      }
      for(int j=i; j<n; j++){
        result.push_back(gamma(distance(&(coords->at(i)), &(coords->at(j)))));
      }
      result.push_back(1.0);
    }
    for(int i=0; i<n; i++){
      result.push_back(1.0);
    }
    result.push_back(0.0);
    return(result);
  }

  class ESI_Kriging: public ESI {
    protected:
      int variogram_model; // 1:Spherical 2:Exponetial 3:Cubic 4:Gaussian
      float nugget, range;

      std::function<float(float)> variogram(int m, float n, float r){
        float c = (1.0-nugget);

        if(m==1){ // Spherical
          return([n,c,r](float d){return(std::min(1.0, std::max(0.0, 1.0 - n - c*(1.5*d/r - 0.5*pow(d/r, 3.0)))));});
        }else if(m==2){ // Exponential
          return([n,c,r](float d){return(std::min(1.0, std::max(0.0, 1.0 - n - c*(1.0-exp(-3.0*d/r)))));});
        }else if(m==3){ // Cubic
          return([n,c,r](float d){return(std::min(1.0, std::max(0.0, 1.0 - n - c*(7.0*pow(d/r, 2.0) - 35.0*pow(d/r, 3.0)/4.0 + 3.5*pow(d/r, 5.0) - 0.75*pow(d/r, 7.0)))));});
        }else if(m==4){ // Gaussian
          return([n,c,r](float d){return(std::min(1.0, std::max(0.0, 1.0 - n - c*(1-exp(-3.0*pow(d/r, 2))))));});
        }else{
          return([](float d){return(d);});
        }
      }

      std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params){
        std::vector<float> result;
        if(params->size()==0){
          for(auto l: *locations_id){
            result.push_back(NAN);
          }
          return(result);
        }
        int n = samples_id->size();
        int m = locations_id->size();
        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        auto sl_locations = slice(locations, locations_id);

        auto right_cov = kriging_right_matrix(&sl_coords, &sl_locations, variogram(variogram_model, nugget, range));
        values->push_back(0.0); // to anulate the mu coeffcicient
        Eigen::Map<Eigen::MatrixXf> v = Eigen::Map<Eigen::MatrixXf>(sl_values.data(), 1, n+1);
        Eigen::Map<Eigen::MatrixXf> b = Eigen::Map<Eigen::MatrixXf>(right_cov.data(), n+1, m);
        Eigen::Map<Eigen::MatrixXf> A_1 = Eigen::Map<Eigen::MatrixXf>(params->data(), n+1, n+1);
        auto weights = A_1*b;
        auto vals = v*weights;
        
/*        Eigen::Map<Eigen::MatrixXf> A = Eigen::Map<Eigen::MatrixXf>(params->data(), n+1, n+1);
        auto x = A.fullPivLu().solve(b);
        auto vals = v*x; */
        result.resize(m);
        Eigen::Map<Eigen::MatrixXf>(&result[0], 1, m) = vals;

        return(result);
      }

      void post_process(){
        int n;
        std::vector<std::vector<float>> leaf_coords;

        for(int i=0; i<mondrian_forest.size(); i++){
          auto mt = mondrian_forest.at(i);
          n = mt->samples_by_leaf.size();
          for(int j=0; j<n; j++){
            leaf_coords.clear();
            for(int k=0; k<mt->samples_by_leaf.at(j).size(); k++){
              leaf_coords.push_back(coords.at(mt->samples_by_leaf.at(j).at(k)));
            }
            mt->leaf_params.at(j) = get_params(&leaf_coords);
          }
        }
      }

      std::vector<float> get_params(std::vector<std::vector<float>> *coords){
        int n = coords->size();
        std::vector<float> result;

        auto klm = kriging_left_matrix(coords, variogram(variogram_model, nugget, range));
        Eigen::Map<Eigen::MatrixXf> LM = Eigen::Map<Eigen::MatrixXf>(klm.data(), n+1, n+1);
        // auto inv = LM.inverse();
        auto inv = LM.completeOrthogonalDecomposition().pseudoInverse();

        result.resize((n+1)*(n+1));
        Eigen::Map<Eigen::MatrixXf>(&result[0], n+1, n+1) = inv;

        return(result);
      }

    public:
      ESI_Kriging(std::vector<std::vector<float>> _coords, std::vector<float> _values, float lambda, int forest_size, std::vector<std::vector<float>> bbox, int _model, float _nugget, float _range, float seed=0):ESI(_coords, _values, lambda, forest_size, bbox, seed){
        variogram_model = _model;
        nugget = _nugget;
        range = _range;
        post_process();
      }
  };
}

#endif
