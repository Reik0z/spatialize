#ifndef _SPTLZ_UTILS_
#define _SPTLZ_UTILS_

#include <vector>
#include <string>
#include <algorithm>
#include <functional>

namespace sptlz{
  template <class T>
  void pprint(std::vector<T> v){
      for(T i: v){
        std::cout << " " << i;
      }
      std::cout << std::endl;
  }

  std::vector<std::vector<int>> get_full_neighboorhood(int d){
    if(d==0){
      std::vector<std::vector<int>> r;
      r.push_back({});
      return(r);
    }else{
      std::vector<std::vector<int>> result = get_full_neighboorhood(d-1);
      std::vector<int> aux;
      int n = result.size();
      for(int i=0; i<n; i++){
        aux = result.at(i);
        aux.push_back(-1);
        result.push_back(aux);
        aux = result.at(i);
        aux.push_back(0);
        result.push_back(aux);
        aux = result.at(i);
        aux.push_back(1);
        result.push_back(aux);
      }
      auto cur = result.begin();
      for(int i=0; i<n; i++){
        cur = result.erase(cur);
      }
      return(result);
    }
  }

  template <class T>
  std::vector<float> grid_search(T *func, std::vector<std::vector<float>> *ranges, std::vector<float> cur, float tol=1e-5){
    int n = ranges->size();
    for(int i=0; i<n; i++){
      if((cur.at(i)<ranges->at(i).at(0))||(ranges->at(i).at(1)<cur.at(i))){
        throw std::runtime_error("starting point outside the bounds");
      }
    }
    float minimum = func->eval(cur), value, diff;
    std::vector<float> new_pos(n), best_candidate;
    auto neighbors = get_full_neighboorhood(n);

    while(true){
      for(auto nb: neighbors){
        bool from_top=false, all_zero=true;
        // refresh new_pos
        for(int i=0; i<n; i++){
          if(nb.at(i)!=0){
            all_zero=false;
          }
          new_pos.at(i) = cur.at(i)+nb.at(i)*ranges->at(i).at(2);
          if((new_pos.at(i) < ranges->at(i).at(0)) || (ranges->at(i).at(1) < new_pos.at(i))){
            from_top = true;
            break;
          }
        }
// std::cout << "DOS " << std::endl;
        // if new position is in place and not all zero (no movement, same point) calculate and refresh best candidate
        if(!from_top && !all_zero){
          value = func->eval(new_pos);
          if(best_candidate.size()==0 || value<best_candidate.at(3)){
            best_candidate = {};
            for(int i=0; i<n; i++){
              best_candidate.push_back(new_pos.at(i));
            }
            best_candidate.push_back(value);
          }
        }
      }
// std::cout << "TRES " << std::endl;
      // difference
      diff = minimum - best_candidate.at(n);
      if(diff>0){
        // if lower, refresh
        minimum = best_candidate.at(n);
        cur = {};
        for(int i=0; i<n; i++){
          cur.push_back(best_candidate.at(i));
        }
        // if close enough, leave
        if(std::abs(diff)<tol){
          break;
        }
      }else{
        // not better than current, so local minimum
        break;
      }
// std::cout << "CUATRO " << std::endl;
    }

    return(cur);
  }

  template <class T>
  std::vector<T> as_1d_array(std::vector<std::vector<T>> *arr, std::vector<int> idxs){
    std::vector<T> result;

    for(auto &record : *arr){
      for(size_t i=0;i<idxs.size();i++){
        result.push_back(record[idxs[i]]);
      }
    }

    return(result);
  }

  template <class T>
  std::vector<T> as_1d_array(std::vector<std::vector<T>> *arr){
    std::vector<int> idxs = {};

    for(size_t i=0;i<arr->at(0).size();i++){
      idxs.push_back(i);
    }

    return(as_1d_array(arr, idxs));
  }

  float distance(std::vector<float> *p1, std::vector<float> *p2){
    float c = 0.0;
    for(size_t i=0; i<p1->size(); i++){
      c += std::pow(p1->at(i)-p2->at(i), 2.0);
    }
    return(std::pow(c, 0.5));
  }

  std::vector<float> distances(std::vector<std::vector<float>> *coords, size_t j){
    std::vector<float> result;

    for(size_t i=0; i<coords->size(); i++){
      if(i!=j){
        result.push_back(distance(&(coords->at(j)), &(coords->at(i))));
      }
    }
    return(result);
  }

  std::vector<float> get_centroid(std::vector<std::vector<float>> *coords){
    std::vector<float> result(coords->at(0).size());
    int n = 0;

    for(size_t i=0; i<coords->size(); i++){
      for(size_t j=0; j<result.size(); j++){
        result.at(j) = result.at(j)+coords->at(i).at(j);
      }
      n++;
    }

    for(size_t j=0; j<result.size(); j++){
      result.at(j) = result.at(j)/n;
    }

    return(result);
  }

  std::vector<std::vector<float>> transform(const float *coords, const float *params, const float *centroid, int n, int d){
    std::vector<std::vector<float>> tr_coords;

    if (d==2){
      float r1 = params[1]*cos(params[0]*3.141592/180.0), r2 = params[1]*sin(params[0]*3.141592/180.0), r3 = -sin(params[0]*3.141592/180.0), r4 = cos(params[0]*3.141592/180.0);
      for(int i=0; i<2*n; i++){
        tr_coords.push_back({
          r1*(coords[2*i]-centroid[0])+r2*(coords[2*i+1]-centroid[1]),
          r3*(coords[2*i]-centroid[0])+r4*(coords[2*i+1]-centroid[1])
        });
      }
    }else if (d==3){
      float ca = cos(params[0]*3.141592/180.0), sa = sin(params[0]*3.141592/180.0), cb = cos(params[1]*3.141592/180.0), sb = sin(params[1]*3.141592/180.0), cc = cos(params[2]*3.141592/180.0), sc = sin(params[2]*3.141592/180.0);
      float r1 = ca*cb, r2 = ca*sb*sc-sa*cc, r3 = ca*sb*cc+sa*sc, r4 = sa*cb, r5 = sa*sb*sc+ca*cc, r6 = sa*sb*cc-ca*sc, r7 = -sb, r8 = cb*sc, r9 = cb*cc;
      r4 *= params[3]; r5 *= params[3]; r6 *= params[3];
      r7 *= params[4]; r8 *= params[4]; r9 *= params[4];
      for(int i=0; i<2*n; i++){
        tr_coords.push_back({
          r1*(coords[3*i]-centroid[0])+r2*(coords[3*i+1]-centroid[1])+r3*(coords[3*i+2]-centroid[2]),
          r4*(coords[3*i]-centroid[0])+r5*(coords[3*i+1]-centroid[1])+r6*(coords[3*i+2]-centroid[2]),
          r7*(coords[3*i]-centroid[0])+r8*(coords[3*i+1]-centroid[1])+r9*(coords[3*i+2]-centroid[2])
        });
      }
    }
    return(tr_coords);
  }

  std::vector<std::vector<float>> transform(std::vector<std::vector<float>> *coords, std::vector<float> *params, std::vector<float> *centroid){
    auto coords_1d = as_1d_array(coords);
    return(transform(coords_1d.data(), params->data(), centroid->data(), coords->size(), centroid->size()));
  }

  template <class T>
  std::vector<T> slice(std::vector<T> *arr, std::vector<int> *idxs){
    std::vector<T> result; 
    for(int i : *idxs){
      result.push_back(arr->at(i));
    }
    return(result);
  }

  template <class T>
  std::vector<T> slice_from(std::vector<T> *arr, int idx){
    std::vector<T> result;
    for(int i=idx; i<arr->size(); i++){
      result.push_back(arr->at(i));
    }
    return(result);
  }

  template <class T>
  std::vector<T> slice_to(std::vector<T> *arr, int idx){
    std::vector<T> result;
    for(int i=0; i<idx; i++){
      result.push_back(arr->at(i));
    }
    return(result);
  }

  template <class T>
  std::vector<T> slice_from_to(std::vector<T> *arr, int from, int to){
    std::vector<T> result;
    for(int i=from; i<to; i++){
      result.push_back(arr->at(i));
    }
    return(result);
  }

  template <class T>
  std::vector<T> slice_drop_idx(std::vector<T> *arr, size_t idx){
    std::vector<T> result;
    for(size_t i=0; i<arr->size(); i++){
      if(i!=idx){
        result.push_back(arr->at(i));
      }
    }
    return(result);
  }

  std::vector<int> get_folds(int n, int k, float seed){
    std::vector<int> result(n);
    std::mt19937 my_rand(seed);
    std::uniform_real_distribution<float> uni_float(0, 1);
    std::vector<std::pair<int, float>> permutation;

    for(int i=0; i<n; i++){
      permutation.push_back(std::make_pair(i,uni_float(my_rand)));
    }
    std::sort(permutation.begin(), permutation.end(), [](auto a, auto b){return(a.second<b.second);});
    for(int i=0; i<n; i++){
      result.at(i) = permutation.at(i).first*k/n;
    }

    return(result);
  }

  template <class T>
  std::pair<std::vector<T>, std::vector<T>> divide_by_predicate(std::vector<T> *arr, std::function<bool(T *)> pred){
    std::vector<T> result1;
    std::vector<T> result2;

    for(auto record: *arr){
      if(pred(&record)){
        result1.push_back(record);
      }else{
        result2.push_back(record);
      }
    }
    return(std::make_pair(result1, result2));
  }

  template <class T>
  std::pair<std::vector<int>, std::vector<int>> indexes_by_predicate(std::vector<T> *arr, std::function<bool(T *)> pred){
    std::vector<int> result1;
    std::vector<int> result2;

    for(size_t i=0; i<arr->size(); i++){
      if(pred(&(arr->at(i)))){
        result1.push_back(i);
      }else{
        result2.push_back(i);
      }
    }
    return(std::make_pair(result1, result2));
  }
}

#endif
