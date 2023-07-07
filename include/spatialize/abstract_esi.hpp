#ifndef _SPTLZ_ESI_
#define _SPTLZ_ESI_

#include <sstream>
#include <random>
#include <queue>
#include <string>
#include <stdexcept>
#include "utils.hpp"

namespace sptlz{
	std::string IND_STEP = "    ";

	std::string bbox_to_json(std::vector<std::vector<float>> bbox, std::string indent){
		std::vector<std::string> axis = {"X", "Y", "Z"};
		std::stringstream s;
		for(size_t i=0; i<bbox.size(); i++){
			if(i==0){
				s << "{" << std::endl;
			}
			s << indent << IND_STEP << "\"" << axis.at(i) << "\": [" << bbox.at(i).at(0) << ", " << bbox.at(i).at(1) << "]";

			if(i==bbox.size()-1){
				s  << std::endl << indent << "}";
			}else{
				s << "," << std::endl;
			}
		}
		return(s.str());
	}

	std::vector<std::vector<float>> samples_coords_bbox(std::vector<std::vector<float>> *coords){
		std::vector<std::vector<float>> bbox;

		for(size_t i=0; i<coords->at(0).size(); i++){

			bbox.push_back({coords->at(0)[i], coords->at(0)[i]});
		}

		for(size_t i=0; i<coords->size(); i++){
			for(size_t j=0; j<coords->at(i).size(); j++){
				if(coords->at(i).at(j) < bbox.at(j).at(0)){
					bbox.at(j).at(0) = coords->at(i).at(j);
				}
				if(bbox.at(j).at(1) < coords->at(i).at(j)){
					bbox.at(j).at(1) = coords->at(i).at(j);
				}
			}
		}

		return(bbox);
	}

	float bbox_sum_interval(std::vector<std::vector<float>> bbox){
		float c = 0.0;
		for(std::vector<float> axis : bbox){
			c += (axis[1]-axis[0]);
		}
		return(c);
	}

	class MondrianNode {
		public:
			int leaf_id;
			std::vector<std::vector<float>> bbox;
			float tau, cut;
			int height, axis;
			MondrianNode* left;
			MondrianNode* right;

			MondrianNode(std::vector<std::vector<float>> _bbox, float _tau, int _height){
				bbox = _bbox;
				tau = _tau;
				height = _height;
				left = NULL;
				right = NULL;
				leaf_id = -1;
			}

			int search_leaf(std::vector<float> point){
				if(leaf_id<0){
					if(point.at(axis) < cut){
						return(left->search_leaf(point));
					}else{
						return(right->search_leaf(point));
					}
				}else{
					return(leaf_id);
				}
			}

			std::string to_json(std::string indent){
				std::stringstream s;
				s << "{" << std::endl;
				s << indent << IND_STEP << "\"bbox\": " <<  bbox_to_json(bbox, indent+IND_STEP);
				if(leaf_id<0){
					s << ",";
				}
				s << std::endl;
				if(left != NULL){
					s << indent << IND_STEP << "\"left\": " << left->to_json(indent+IND_STEP);
					if(right != NULL){
						s << ",";
					}
					s << std::endl;
				}
				if(right != NULL){
					s << indent << IND_STEP << "\"right\": " << right->to_json(indent+IND_STEP) << std::endl;
				}
				s << indent << "}";
				return(s.str());
			}
	};

	class MondrianTree {
		public:
			MondrianNode* root;
			std::vector<MondrianNode*> leaves;
			std::vector<int> leaf_for_sample;
			std::vector<std::vector<int>> samples_by_leaf;
			std::vector<std::vector<float>> leaf_params;
			int ndim;
			float* coords;
			float* values;

			MondrianTree(std::vector<std::vector<float>> *coords, float lambda, std::vector<std::vector<float>> bbox, float seed=0){
				ndim = (int) bbox.size();
				std::mt19937 my_rand(seed);
				std::uniform_int_distribution<int> uni_int(0, ndim-1);
				std::uniform_real_distribution<float> uni_float(0, 1);
				std::exponential_distribution<float> exp_float;
				std::queue<MondrianNode*> bft;

				// assign root node
				MondrianNode* cur_node = new MondrianNode(bbox, 0, 0);
				MondrianNode* aux_node;
				std::vector<std::vector<float>> cur_bbox, aux_bbox;
				float aux_value;

				root = cur_node;

				if(lambda>0){
					// there is lifetime so it COULD be splitted
					bft.push(cur_node);
				}

				// when there are nodes to split
				while(!bft.empty()){
					// get the node
					cur_node = bft.front();
					bft.pop();

					// if node should be splitted, then do it
					if (cur_node->tau < lambda){
						// get the bounding box for the node
						cur_bbox = cur_node->bbox;
						// select the component (axis) to make the cut
						cur_node->axis = uni_int(my_rand);
						// the cut will MIN + (MAX-MIN) * random(0,1)
						cur_node->cut = cur_bbox[cur_node->axis][0] + (cur_bbox[cur_node->axis][1] - cur_bbox[cur_node->axis][0]) * uni_float(my_rand); // CHANGE FOR RANDOM_UNI_FLOAT(0,1)

						// LOWER THAN CHILD
						// bbox for child (just changes the upper limit for selected axis)
						aux_bbox = cur_bbox;
						aux_bbox[cur_node->axis][1] = cur_node->cut;
						// get interval for bbox
						aux_value = bbox_sum_interval(aux_bbox);
						// set the lambda parameter for exponential distribution
						exp_float.param(std::exponential_distribution<float>::param_type(aux_value));
						// get the tau for the child
						aux_value = cur_node->tau + exp_float(my_rand);
						// create the new child
						aux_node = new MondrianNode(aux_bbox, aux_value, cur_node->height+1);
						// set a left child
						cur_node->left = aux_node;
						// if it must be splitted, put it in the queue
						if(aux_value < lambda){
							bft.push(aux_node);
						}

						// GREATER THAN CHILD
						// bbox for child (just changes the lower limit for selected axis)
						aux_bbox = cur_bbox;
						aux_bbox[cur_node->axis][0] = cur_node->cut;
						// get interval for bbox
						aux_value = bbox_sum_interval(aux_bbox);
						// set the lambda parameter for exponential distribution
						exp_float.param(std::exponential_distribution<float>::param_type(aux_value));
						// get the tau for the child
						aux_value = cur_node->tau + exp_float(my_rand);
						// create the new child
						aux_node = new MondrianNode(aux_bbox, aux_value, cur_node->height+1);
						// set a right child
						cur_node->right = aux_node;
						// if it must be splitted, put it in the queue
						if(aux_value < lambda){
							bft.push(aux_node);
						}
					}
				}

				// put the root in the queue (it should be empty)
				bft.push(root);
				// visit all nodes
				while(!bft.empty()){
					// get the node
					cur_node = bft.front();
					bft.pop();

					if((cur_node->left==NULL) && (cur_node->right==NULL)){
						cur_node->leaf_id = (int) leaves.size();
						leaves.push_back(cur_node);
						samples_by_leaf.push_back({});
						leaf_params.push_back({});
					}else{
						bft.push(cur_node->left);
						bft.push(cur_node->right);
					}
				}

				// assign samples to leafs and inverse too
				int aux;
				for(size_t i=0; i<coords->size(); i++){
					aux = search_leaf(coords->at(i));
					samples_by_leaf.at(aux).push_back(i);
					leaf_for_sample.push_back(aux);
				}
	  		}

	  		int search_leaf(std::vector<float> point){
	  			auto bbox = root->bbox;
	  			for(size_t i=0; i<point.size(); i++){
	  				if((point.at(i) < bbox.at(i).at(0)) || (bbox.at(i).at(1) < point.at(i))){
	  					return(-1);
	  				}
	  			}
	  			return(root->search_leaf(point));
	  		}

	  		std::string to_json(){
	  			return(root->to_json(""));
	  		}
	};

	class ESI {
		protected:
			std::vector<sptlz::MondrianTree*> mondrian_forest;
			std::vector<std::vector<float>> coords;
			std::vector<float> values;
			std::vector<int> folds;
			std::mt19937 my_rand;

			virtual std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params){
				throw std::runtime_error("must override");
			}

			virtual std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params){
				throw std::runtime_error("must override");
			}

			virtual std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *fold, std::vector<int> *samples_id, std::vector<float> *params){
				throw std::runtime_error("must override");
			}

			virtual void post_process(){}

		public:
			ESI(std::vector<std::vector<float>> _coords, std::vector<float> _values, float lambda, int forest_size, std::vector<std::vector<float>> bbox, float seed=0){
				my_rand = std::mt19937(seed);
				coords = _coords;
				values = _values;
				std::uniform_int_distribution<int> uni_int;

				for(int i=0; i<forest_size; i++){
					mondrian_forest.push_back(new sptlz::MondrianTree(&coords, lambda, bbox, uni_int(my_rand)));
				}
			}

			std::vector<std::vector<float>> estimate(std::vector<std::vector<float>> *locations){
				std::vector<std::vector<float>> results(locations->size());
				std::vector<std::vector<int>> locations_by_leaf;
				int aux;

				for(size_t i=0; i<mondrian_forest.size(); i++){
					// get tree
					auto mt = mondrian_forest.at(i);
					locations_by_leaf = std::vector<std::vector<int>>(mt->leaves.size());

					// join all locations for same leaf
					for(size_t j=0; j<locations->size(); j++){
						aux = mt->search_leaf(locations->at(j));
						locations_by_leaf.at(aux).push_back(j);
					}

					// make estimation by leaf
					for(size_t j=0; j<locations_by_leaf.size(); j++){
						if(mt->samples_by_leaf.at(j).size()==0){
							for(size_t k=0; k<locations_by_leaf.at(j).size(); k++){
								results.at(locations_by_leaf.at(j).at(k)).push_back(NAN);
							}
						}else{
							auto predictions = leaf_estimation(&coords, &values, &(mt->samples_by_leaf.at(j)), locations, &(locations_by_leaf.at(j)), &(mt->leaf_params.at(j)));
							for(size_t k=0; k<locations_by_leaf.at(j).size(); k++){
								results.at(locations_by_leaf.at(j).at(k)).push_back(predictions.at(k));
							}
						}
					}
				}
				return(results);
			}

			std::vector<std::vector<float>> leave_one_out(){
				std::vector<std::vector<float>> results(coords.size());

				for(size_t i=0; i<mondrian_forest.size(); i++){
					// get tree
					auto mt = mondrian_forest.at(i);

					// make loo by leaf
					for(size_t j=0; j<mt->samples_by_leaf.size(); j++){
						if(mt->samples_by_leaf.at(j).size()!=0){
							auto predictions = leaf_loo(&coords, &values, &(mt->samples_by_leaf.at(j)), &(mt->leaf_params.at(j)));
							for(size_t k=0; k<mt->samples_by_leaf.at(j).size(); k++){
								results.at(mt->samples_by_leaf.at(j).at(k)).push_back(predictions.at(k));
							}
						}
					}
				}
				return(results);
			}

			std::vector<std::vector<float>> k_fold(int k){
				std::uniform_real_distribution<float> uni_float;
				auto folds = get_folds(values.size(), k, uni_float(my_rand));
				std::vector<std::vector<float>> results(coords.size());

				for(size_t i=0; i<mondrian_forest.size(); i++){
					// get tree
					auto mt = mondrian_forest.at(i);

					// make kfold by leaf
					for(size_t j=0; j<mt->samples_by_leaf.size(); j++){
						if(mt->samples_by_leaf.at(j).size()!=0){
							auto predictions = leaf_kfold(k, &coords, &values, &folds, &(mt->samples_by_leaf.at(j)), &(mt->leaf_params.at(j)));
							for(size_t k=0; k<mt->samples_by_leaf.at(j).size(); k++){
								results.at(mt->samples_by_leaf.at(j).at(k)).push_back(predictions.at(k));
							}
						}
					}
				}
				return(results);
			}
	};
}

#endif
