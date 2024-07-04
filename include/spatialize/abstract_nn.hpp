#ifndef _SPTLZ_NN_
#define _SPTLZ_NN_

#include <vector>
#include <random>
#include <functional>
#include "kdtree.hpp"
#include "utils.hpp"

namespace sptlz{

	class NN{
		protected:
			int n_samples, n_dims;
			float radius;
			std::vector<std::vector<float>> coords;
			std::vector<float> values;
			std::vector<float> search_params;
      sptlz::KDTree<float> *kdt;

      virtual float estimate_point(std::pair<std::vector<float>, std::vector<int>> *nbs, std::vector<float> *pt){
				throw std::runtime_error("must override");
      }

      virtual float estimate_loo(std::pair<std::vector<float>, std::vector<int>> *nbs, size_t i){
				throw std::runtime_error("must override");
      }

      virtual float estimate_kfold(std::pair<std::vector<float>, std::vector<int>> *nbs, int i, std::vector<int> *folds){
				throw std::runtime_error("must override");
      }

		public:
			NN(std::vector<std::vector<float>> _coords, std::vector<float> _values, std::vector<float> _search_params){
				this->n_samples = _coords.size();
				this->n_dims = _coords.at(0).size();
				this->coords = _coords;
				this->values = _values;
        this->search_params = _search_params; // TODO: for anisotropic searches, scale, rotate and set radius=1
        this->radius = search_params.at(0);

        this->kdt = new sptlz::KDTree<float>(&(this->coords));
			}

			~NN(){
        if(this->kdt != NULL){
          delete(this->kdt);
        }
      }

			std::vector<float> estimate(std::vector<std::vector<float>> *locations, std::function<int(std::string)> visitor){
				std::stringstream json;
				std::vector<float> result;
				float value;
				int n = locations->size();

				// {"message": {"text": "<the log text>", "level": "<DEBUG|INFO|WARNING|ERROR|CRITICAL>"}}
				json.str("");
				json << "{\"message\": { \"text\":\"" << "[C++|NN->estimate] computing estimates" << "\", \"level\":\"" << "DEBUG" <<"\"}}";
				visitor(json.str());

				// {"progress": {"init": <total expected count>, "step": <increment step>}}
				json.str("");
				json << "{\"progress\": { \"init\":" << n << ", \"step\":" << 1 <<"}}";
				visitor(json.str());

                for(int i=0; i<n; i++){
                  // {"progress": {"token": <value>}}
                  json.str("");
                  json << "{\"progress\": {\"token\":" << 100.0*(i+1.0)/n << "}}";
                  if (PyErr_CheckSignals() != 0)  // to allow ctrl-c from user
                      exit(0);
                  visitor(json.str());

                  auto nbs = this->kdt->query_ball(&(locations->at(i)), radius, 2.0);
                  value = this->estimate_point(&nbs, &(locations->at(i)));
                  result.push_back(value);
                }

				// {"progress": "done"}
				json.str("");
				json << "{\"progress\": \"done\"}";
				visitor(json.str());

				return(result);
			}

			std::vector<float> leave_one_out(std::function<int(std::string)> visitor){
				std::stringstream json;
				std::vector<float> result;
				float value;
				int n = coords.size();

				// {"message": {"text": "<the log text>", "level": "<DEBUG|INFO|WARNING|ERROR|CRITICAL>"}}
				json.str("");
				json << "{\"message\": { \"text\":\"" << "[C++|NN->leave_one_out] computing estimates" << "\", \"level\":\"" << "DEBUG" <<"\"}}";
				visitor(json.str());

				// {"progress": {"init": <total expected count>, "step": <increment step>}}
				json.str("");
				json << "{\"progress\": { \"init\":" << n << ", \"step\":" << 1 <<"}}";
				visitor(json.str());

                for(int i=0; i<n; i++){
                  // {"progress": {"token": <value>}}
                  json.str("");
                  json << "{\"progress\": {\"token\":" << 100.0*(i+1.0)/n << "}}";
                  if (PyErr_CheckSignals() != 0)  // to allow ctrl-c from user
                      exit(0);
                  visitor(json.str());
                  auto nbs = this->kdt->query_ball(&(coords.at(i)), radius, 2.0);
                  value = this->estimate_loo(&nbs, i);
                  result.push_back(value);
                }

				// {"progress": "done"}
				json.str("");
				json << "{\"progress\": \"done\"}";
				visitor(json.str());

				return(result);
			}

			std::vector<float> k_fold(int k, std::function<int(std::string)> visitor, int seed=206936){
				std::stringstream json;
				std::uniform_real_distribution<float> uni_float;
				std::mt19937 my_rand(seed);
				std::vector<float> result;
				auto folds = get_folds(values.size(), k, uni_float(my_rand));
				float value;
				int n = coords.size();

				// {"message": {"text": "<the log text>", "level": "<DEBUG|INFO|WARNING|ERROR|CRITICAL>"}}
				json.str("");
				json << "{\"message\": { \"text\":\"" << "[C++|NN->k_fold] computing estimates" << "\", \"level\":\"" << "DEBUG" <<"\"}}";
				visitor(json.str());

				// {"progress": {"init": <total expected count>, "step": <increment step>}}
				json.str("");
				json << "{\"progress\": { \"init\":" << n << ", \"step\":" << 1 <<"}}";
				visitor(json.str());

                for(int i=0; i<n; i++){
                  // {"progress": {"token": <value>}}
                  json.str("");
                  json << "{\"progress\": {\"token\":" << 100.0*(i+1.0)/n << "}}";
                  if (PyErr_CheckSignals() != 0)  // to allow ctrl-c from user
                      exit(0);
                  visitor(json.str());
                  auto nbs = this->kdt->query_ball(&(coords.at(i)), radius, 2.0);
                  value = this->estimate_kfold(&nbs, i, &folds);
                  result.push_back(value);
                }

				// {"progress": "done"}
				json.str("");
				json << "{\"progress\": \"done\"}";
				visitor(json.str());

				return(result);
			}

	};
}

#endif
