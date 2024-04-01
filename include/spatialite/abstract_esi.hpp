#ifndef _SPTLZ_ESI_
#define _SPTLZ_ESI_

#include <string>
#include <random>
#include <queue>
#include <sstream>
#include <functional>
#include <vector>
#include <utility>
#include <stdexcept>
#include <sqlite3.h>
#include "utils.hpp"

namespace sptlz{

	std::vector<float> coords_bbox(std::vector<std::vector<float>> *coords){
		std::vector<float> bbox;
		int n=coords->size(), m=coords->at(0).size();

		for(int i=0; i<m; i++){
			bbox.push_back(coords->at(0)[i]);
			bbox.push_back(coords->at(0)[i]);
		}

		for(int i=0; i<n; i++){
			for(int j=0; j<m; j++){
				if(coords->at(i).at(j) < bbox.at(2*j)){
					bbox.at(2*j) = coords->at(i).at(j);
				}
				if(bbox.at(2*j+1) < coords->at(i).at(j)){
					bbox.at(2*j+1) = coords->at(i).at(j);
				}
			}
		}

		return(bbox);
	}

	float bbox_sum_interval(std::vector<float> bbox){
		int i, n=bbox.size()/2;
		float c = 0.0;
		for(i=0; i<n; i++){
			c += (bbox.at(2*i+1)-bbox.at(2*i));
		}
		return(c);
	}

	std::string get_esi_type(std::string path){
		sqlite3 *db;
		sqlite3_stmt *stmt;
		char* err_msg = 0;

		int rc = sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, NULL);

		sqlite3_exec(db, "PRAGMA synchronous = OFF", NULL, NULL, &err_msg);
        sqlite3_exec(db, "PRAGMA journal_mode = MEMORY", NULL, NULL, &err_msg);

		// ask if database was open ok
		if(rc) {
			std::stringstream msg("");
			msg << "[get_esi_type|1] Can't open database: ";
			msg << sqlite3_errmsg(db);
			sqlite3_close_v2(db);
			throw std::runtime_error(msg.str());
		}

		std::string result ="", query = "SELECT type, value FROM params WHERE key='esi_type';";
		rc = sqlite3_prepare_v3(db, query.c_str(), query.length(), 0, &stmt, NULL);

		// compilation not ok
		if(rc) {
			std::stringstream msg("");
			msg << "[get_esi_type|2] Cannot compile statememt: ";
			msg << sqlite3_errmsg(db);
			sqlite3_close_v2(db);
			throw std::runtime_error(msg.str());
		}
		// get value
		rc = sqlite3_step(stmt);
		if(rc == SQLITE_ROW){
			result = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
			if(result!="text"){
				std::stringstream msg("");
				msg << "[get_esi_type|3] The value for key 'esi_type' was stored as " << result;
				msg << sqlite3_errmsg(db);
				sqlite3_close_v2(db);
				throw std::runtime_error(msg.str());
			}else{
				result = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
			}
		}

		sqlite3_finalize(stmt);
		sqlite3_close_v2(db);

		return(result);
	}

	class ESI{
		protected:

			sqlite3 *db;
			std::vector<std::vector<float>> coords, extras;
			std::vector<float> values;

			void open_database(std::string path){
			    char* err_msg = 0;
				int rc = sqlite3_open_v2(path.c_str(), &(this->db), SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, NULL);

				sqlite3_exec(this->db, "PRAGMA synchronous = OFF", NULL, NULL, &err_msg);
                sqlite3_exec(this->db, "PRAGMA journal_mode = MEMORY", NULL, NULL, &err_msg);


				// ask if database was open ok
				if(rc) {
					std::stringstream msg("");
					msg << "[open_database] Can't open database: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}
			}

			void create_table(std::string table_name, std::string sql){
				// ask for tables
				sqlite3_stmt *stmt;
				char* err_msg = 0;
				std::string query = "SELECT name FROM sqlite_master WHERE type='table' AND name='"+table_name+"';";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[create_table|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				// verify if table already exist
				rc = sqlite3_step(stmt);
				if(rc == SQLITE_ROW){
					// build query to create the table
					std::stringstream aux("");
					aux << "BEGIN TRANSACTION;";
					aux << "DELETE FROM '" + table_name + "';";
					aux << "COMMIT;";
					query = aux.str();
					rc = sqlite3_exec(this->db, query.c_str(), NULL, NULL, &err_msg);

					// verify removal was ok
					if(rc) {
						std::stringstream msg("");
						msg << "[create_table|2] Cannot delete records: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
				}else{
					// build query to create the table
					std::stringstream aux("");
					aux << "BEGIN TRANSACTION;";
					aux << sql.c_str();
					aux << "COMMIT;";
					query = aux.str();
					rc = sqlite3_exec(this->db, query.c_str(), NULL, NULL, &err_msg);

					// verify creation was ok
					if(rc) {
						std::stringstream msg("");
						msg << "[create_table|3] Cannot create table: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
				}

				sqlite3_finalize(stmt);
			}

			void begin_transaction(){
				// begin transaction
				char* err_msg = 0;
				int rc = sqlite3_exec(this->db, "BEGIN TRANSACTION;", NULL, NULL, &err_msg);
				if(rc) {
					std::stringstream msg("");
					msg << "[begin|3] Cannot execute BEGIN TRANSACTION: ";
					msg << err_msg;
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}
			}

			void end_transaction(){
				// end transaction
				char* err_msg = 0;
				int rc = sqlite3_exec(this->db, "COMMIT;", NULL, NULL, &err_msg);
				if(rc) {
					std::stringstream msg("");
					msg << "[end_transaction] Cannot execute COMMIT: ";
					msg << err_msg;
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}
			}

			void add_samples(std::vector<std::vector<float>> *coords, std::vector<float> *values){
				sqlite3_stmt *stmt1, *stmt2;
				int i, j, id, n =coords->at(0).size(), m=coords->size();

				std::string query = "INSERT INTO 'positions' (id, sample_id, axis, coord) VALUES(?,?,?,?);";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt1, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[add_samples|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				// compilation not ok
				query = "INSERT INTO 'samples' (id, value) VALUES(?,?);";
				rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt2, NULL);
				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[add_samples|2] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				this->begin_transaction();
					for(j=0;j<m;j++){
						for(i=0;i<n;i++){
							// reset before bind
							sqlite3_reset(stmt1);
							// set coords
							id = j*n+i;
							rc = sqlite3_bind_int(stmt1, 1, id);
							if(rc) {
								std::stringstream msg("");
								msg << "[add_samples|3] Cannot bind value [id," << j << "," << i << "] to statememt: ";
								msg << sqlite3_errmsg(this->db);
								sqlite3_close_v2(this->db);
								throw std::runtime_error(msg.str());
							}
							rc = sqlite3_bind_int(stmt1, 2, j);
							if(rc) {
								std::stringstream msg("");
								msg << "[add_samples|4] Cannot bind value [sample_id," << j << "," << i << "] to statememt: ";
								msg << sqlite3_errmsg(this->db);
								sqlite3_close_v2(this->db);
								throw std::runtime_error(msg.str());
							}
							rc = sqlite3_bind_int(stmt1, 3, i);
							if(rc) {
								std::stringstream msg("");
								msg << "[add_samples|5] Cannot bind value [axis," << j << "," << i << "] to statememt: ";
								msg << sqlite3_errmsg(this->db);
								sqlite3_close_v2(this->db);
								throw std::runtime_error(msg.str());
							}
							rc = sqlite3_bind_double(stmt1, 4, coords->at(j).at(i));
							if(rc) {
								std::stringstream msg("");
								msg << "[add_samples|6] Cannot bind value [coord," << j << "," << i << "] to statememt: ";
								msg << sqlite3_errmsg(this->db);
								sqlite3_close_v2(this->db);
								throw std::runtime_error(msg.str());
							}
							// the insertion
							rc = sqlite3_step(stmt1);
							if(rc != SQLITE_DONE){
								std::stringstream msg("");
								msg << "[add_samples|7] Insertion fails[" << j << "," << i << "]: ";
								msg << sqlite3_errmsg(this->db);
								sqlite3_close_v2(this->db);
								throw std::runtime_error(msg.str());
							}
						}
						// reset before bind
						sqlite3_reset(stmt2);
						rc = sqlite3_bind_int(stmt2, 1, j);
						if(rc) {
							std::stringstream msg("");
							msg << "[add_samples|8] Cannot bind value [id," << j << "] to statememt: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
						rc = sqlite3_bind_double(stmt2, 2, values->at(j));
						if(rc) {
							std::stringstream msg("");
							msg << "[add_samples|9] Cannot bind value [value," << j << "] to statememt: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
						// the insertion
						rc = sqlite3_step(stmt2);
						if(rc != SQLITE_DONE){
							std::stringstream msg("");
							msg << "[add_samples|10] Insertion fails[" << j  << "]: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
					}
				this-> end_transaction();

				sqlite3_finalize(stmt1);
				sqlite3_finalize(stmt2);
			}

			void add_root(int id, int leaf_id){
				sqlite3_stmt *stmt;

				std::string query = "INSERT INTO 'roots' (id, leaf_id) VALUES(?,?);";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[add_root|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				this->begin_transaction();
					// set values
					rc = sqlite3_bind_int(stmt, 1, id);
					if(rc) {
						std::stringstream msg("");
						msg << "[add_root|2] Cannot bind value [id] to statememt: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
					rc = sqlite3_bind_int(stmt, 2, leaf_id);
					if(rc) {
						std::stringstream msg("");
						msg << "[add_root|3] Cannot bind value [leaf_id] to statememt: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}

					// the insertion
					rc = sqlite3_step(stmt);
					if(rc != SQLITE_DONE){
						std::stringstream msg("");
						msg << "[add_root|4] Insertion fails: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
				this-> end_transaction();

				sqlite3_finalize(stmt);
			}

			void add_leaf(int tree_id, std::vector<float> *node){
				sqlite3_stmt *stmt;

				std::string query = "INSERT INTO 'leaves' (id, tree_id, tau, cut, height, axis, lower, greater) VALUES(?,?,?,?,?,?,?,?);";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[add_leaf|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				this->begin_transaction();
					// build insertion
					rc = sqlite3_bind_int(stmt, 1, (int)node->at(0));
					if(rc) {
						std::stringstream msg("");
						msg << "[add_leaf|2] Cannot bind value [id] to statememt: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
					rc = sqlite3_bind_int(stmt, 2, tree_id);
					if(rc) {
						std::stringstream msg("");
						msg << "[add_leaf|3] Cannot bind value [tree_id] to statememt: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
					rc = sqlite3_bind_double(stmt, 3, node->at(1));
					if(rc) {
						std::stringstream msg("");
						msg << "[add_leaf|4] Cannot bind value [tau] to statememt: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
					rc = sqlite3_bind_double(stmt, 4, node->at(2));
					if(rc) {
						std::stringstream msg("");
						msg << "[add_leaf|5] Cannot bind value [cut] to statememt: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
					rc = sqlite3_bind_int(stmt, 5, (int)node->at(3));
					if(rc) {
						std::stringstream msg("");
						msg << "[add_leaf|6] Cannot bind value [height] to statememt: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
					rc = sqlite3_bind_int(stmt, 6, (int)node->at(4));
					if(rc) {
						std::stringstream msg("");
						msg << "[add_leaf|7] Cannot bind value [axis] to statememt: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
					if(node->at(5)>=0){
						rc = sqlite3_bind_int(stmt, 7, (int)node->at(5));
						if(rc) {
							std::stringstream msg("");
							msg << "[add_leaf|8] Cannot bind value [lower] to statememt: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
					}
					if(node->at(6)>=0){
						rc = sqlite3_bind_int(stmt, 8, (int)node->at(6));
						if(rc) {
							std::stringstream msg("");
							msg << "[add_leaf|9] Cannot bind value [greater] to statememt: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
					}
					// the insertion
					rc = sqlite3_step(stmt);
					if(rc != SQLITE_DONE){
						std::stringstream msg("");
						msg << "[add_leaf|10] Insertion fails: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
				this->end_transaction();

				sqlite3_finalize(stmt);
			}

			void add_bbox(int tree_id, int leaf_id, std::vector<float> *limits){
				sqlite3_stmt *stmt;
				int i, id, n = limits->size()/2;

				std::string query = "INSERT INTO 'bboxes' (id, tree_id, leaf_id, axis, lower_bound, upper_bound) VALUES(?,?,?,?,?,?);";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[add_bbox|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				this->begin_transaction();
					// insertion loop
					for(i=0;i<n;i++){
						// reset before bind
						sqlite3_reset(stmt);
						// set values
						id = leaf_id*limits->size()/2+i;

						rc = sqlite3_bind_int(stmt, 1, id);
						if(rc) {
							std::stringstream msg("");
							msg << "[add_bbox|2] Cannot bind value [id," << i << "] to statememt: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
						rc = sqlite3_bind_int(stmt, 2, tree_id);
						if(rc) {
							std::stringstream msg("");
							msg << "[add_bbox|3] Cannot bind value [tree_id," << i << "] to statememt: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
						rc = sqlite3_bind_int(stmt, 3, leaf_id);
						if(rc) {
							std::stringstream msg("");
							msg << "[add_bbox|4] Cannot bind value [leaf_id," << i << "] to statememt: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
						rc = sqlite3_bind_int(stmt, 4, i);
						if(rc) {
							std::stringstream msg("");
							msg << "[add_bbox|5] Cannot bind value [axis," << i << "] to statememt: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
						rc = sqlite3_bind_double(stmt, 5, limits->at(2*i));
						if(rc) {
							std::stringstream msg("");
							msg << "[add_bbox|6] Cannot bind value [lower_bound," << i << "] to statememt: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
						rc = sqlite3_bind_double(stmt, 6, limits->at(2*i+1));
						if(rc) {
							std::stringstream msg("");
							msg << "[add_bbox|7] Cannot bind value [upper_bound," << i << "] to statememt: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
						// the insertion
						rc = sqlite3_step(stmt);
						if(rc != SQLITE_DONE){
							std::stringstream msg("");
							msg << "[add_bbox|8] Insertion fails[" << i << "]: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
					}
				this->end_transaction();

				sqlite3_finalize(stmt);
			}

			void clear_queries(){
				sqlite3_stmt *stmt1, *stmt2;

				std::string query = "DELETE FROM 'queries';";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt1, NULL);
				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[clear_queries|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				query = "DELETE FROM 'queries_matches';";
				rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt2, NULL);
				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[clear_queries|2] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				this->begin_transaction();
					// delete stuff
					rc = sqlite3_step(stmt1);
					if(rc != SQLITE_DONE){
						std::stringstream msg("");
						msg << "[clear_queries|3] Removal fails: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
					// delete stuff
					rc = sqlite3_step(stmt2);
					if(rc != SQLITE_DONE){
						std::stringstream msg("");
						msg << "[clear_queries|4] Removal fails: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
				this-> end_transaction();

				sqlite3_finalize(stmt1);
				sqlite3_finalize(stmt2);
			}

			void add_queries(std::vector<std::vector<float>> *coords){
				sqlite3_stmt *stmt;
				int i, j, id, n=coords->at(0).size(), m=coords->size();

				std::string query = "INSERT INTO 'queries' (id, query_id, axis, coord) VALUES(?,?,?,?);";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[add_queries|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				this->begin_transaction();
					for(j=0;j<m;j++){
						for(i=0;i<n;i++){
							// reset before bind
							sqlite3_reset(stmt);
							// set coords
							id = j*n+i;
							rc = sqlite3_bind_int(stmt, 1, id);
							if(rc) {
								std::stringstream msg("");
								msg << "[add_queries|3] Cannot bind value [id," << j << "," << i << "] to statememt: ";
								msg << sqlite3_errmsg(this->db);
								sqlite3_close_v2(this->db);
								throw std::runtime_error(msg.str());
							}
							rc = sqlite3_bind_int(stmt, 2, j);
							if(rc) {
								std::stringstream msg("");
								msg << "[add_queries|3] Cannot bind value [query_id," << j << "," << i << "] to statememt: ";
								msg << sqlite3_errmsg(this->db);
								sqlite3_close_v2(this->db);
								throw std::runtime_error(msg.str());
							}
							rc = sqlite3_bind_int(stmt, 3, i);
							if(rc) {
								std::stringstream msg("");
								msg << "[add_queries|3] Cannot bind value [axis," << j << "," << i << "] to statememt: ";
								msg << sqlite3_errmsg(this->db);
								sqlite3_close_v2(this->db);
								throw std::runtime_error(msg.str());
							}
							rc = sqlite3_bind_double(stmt, 4, coords->at(j).at(i));
							if(rc) {
								std::stringstream msg("");
								msg << "[add_queries|3] Cannot bind value [coord," << j << "," << i << "] to statememt: ";
								msg << sqlite3_errmsg(this->db);
								sqlite3_close_v2(this->db);
								throw std::runtime_error(msg.str());
							}
							// the insertion
							rc = sqlite3_step(stmt);
							if(rc != SQLITE_DONE){
								std::stringstream msg("");
								msg << "[add_queries|4] Insertion fails[" << j << "," << i << "]: ";
								msg << sqlite3_errmsg(this->db);
								sqlite3_close_v2(this->db);
								throw std::runtime_error(msg.str());
							}
						}
					}
				this->end_transaction();

				sqlite3_finalize(stmt);
			}

			void set_samples_to_leaves(){
				int rc;
				char* err_msg = 0;
				std::string query, sql = ""
					"WITH RECURSIVE get_leaf(leaf_id, tree_id, level, sample_id) AS ( "
						"SELECT roots.leaf_id, roots.id, 0, aux.sample_id FROM (SELECT DISTINCT positions.sample_id FROM positions) aux, roots "
						"UNION ALL "
						"SELECT CASE WHEN positions.coord < leaves.cut THEN lower ELSE greater END, get_leaf.tree_id, get_leaf.level+1, positions.sample_id "
						"FROM positions, leaves, get_leaf "
						"WHERE positions.axis=leaves.axis AND leaves.id=get_leaf.leaf_id AND positions.sample_id=get_leaf.sample_id "
					") "
					"INSERT INTO samples_matches (tree_id, sample_id, leaf_id)"
					"SELECT get_leaf.tree_id, maxs.sample_id, get_leaf.leaf_id AS leaf_id FROM get_leaf, (SELECT sample_id, MAX(level) AS level, tree_id FROM get_leaf GROUP BY sample_id, tree_id) maxs WHERE maxs.sample_id=get_leaf.sample_id AND maxs.tree_id=get_leaf.tree_id AND get_leaf.level=maxs.level ORDER BY maxs.sample_id, get_leaf.tree_id;";

				// build query to create the table
				std::stringstream aux("");
				aux << "BEGIN TRANSACTION;";
				aux << sql.c_str();
				aux << "COMMIT;";
				query = aux.str();
				rc = sqlite3_exec(this->db, query.c_str(), NULL, NULL, &err_msg);

				// verify creation was ok
				if(rc) {
					std::stringstream msg("");
					msg << "[set_samples_to_leaves] Cannot search/insert data: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}
			}

			void set_queries_to_leaves(){
				int rc;
				char* err_msg = 0;
				std::string query, sql = ""
					"WITH RECURSIVE get_leaf(leaf_id, tree_id, level, query_id) AS ( "
						"SELECT roots.leaf_id, roots.id, 0, aux.query_id FROM (SELECT DISTINCT queries.query_id FROM queries) aux, roots "
						"UNION ALL "
						"SELECT CASE WHEN queries.coord < leaves.cut THEN lower ELSE greater END, get_leaf.tree_id, get_leaf.level+1, queries.query_id "
						"FROM queries, leaves, get_leaf "
						"WHERE queries.axis=leaves.axis AND leaves.id=get_leaf.leaf_id AND queries.query_id=get_leaf.query_id "
					") "
					"INSERT INTO queries_matches (tree_id, query_id, leaf_id)"
					"SELECT get_leaf.tree_id, maxs.query_id, get_leaf.leaf_id AS leaf_id FROM get_leaf, (SELECT query_id, MAX(level) AS level, tree_id FROM get_leaf GROUP BY query_id, tree_id) maxs WHERE maxs.query_id=get_leaf.query_id AND maxs.tree_id=get_leaf.tree_id AND get_leaf.level=maxs.level ORDER BY maxs.query_id, get_leaf.tree_id;";

				// build query to create the table
				std::stringstream aux("");
				aux << "BEGIN TRANSACTION;";
				aux << sql.c_str();
				aux << "COMMIT;";
				query = aux.str();
				rc = sqlite3_exec(this->db, query.c_str(), NULL, NULL, &err_msg);

				// verify creation was ok
				if(rc) {
					std::stringstream msg("");
					msg << "[set_queries_to_leaves] Cannot search/insert data: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}
			}

			std::vector<std::pair<int,std::vector<int>>> get_leaves_n_queries(int tree_id){
				std::vector<std::pair<int,std::vector<int>>> result;
				std::vector<int> queries;
				sqlite3_stmt *stmt;
				int leaf_id=-1, aux;

				std::string query = ""
					"SELECT leaf_id, query_id "
					"FROM queries_matches "
					"WHERE tree_id=? "
					"ORDER BY leaf_id ASC;";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[get_leaves_n_queries|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				// set values
				rc = sqlite3_bind_int(stmt, 1, tree_id);
				if(rc) {
					std::stringstream msg("");
					msg << "[get_leaves_n_queries|2] Cannot bind value [tree_id] to statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				while(true){
					rc = sqlite3_step(stmt);
					if(rc == SQLITE_DONE){
						if(0<queries.size()){
							result.push_back(std::make_pair(leaf_id, queries));
						}
						break;
					}else if(rc == SQLITE_ROW){
						aux = sqlite3_column_int(stmt, 0);
						if(aux!=leaf_id){ // new leaf_id
							// store last
							if(leaf_id>-1){
								result.push_back(std::make_pair(leaf_id, queries));
							}
							leaf_id = aux;
							queries = {sqlite3_column_int(stmt, 1)};
						}else{
							queries.push_back(sqlite3_column_int(stmt, 1));
						}
					}else{
						std::stringstream msg("");
						msg << "[get_leaves_n_queries|4] Cannot retrieve data: ";
						msg << sqlite3_errmsg(db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
				}

				sqlite3_finalize(stmt);
				return(result);
			}

			std::vector<int> get_samples_id(int leaf_id){
				std::vector<int> result;
				sqlite3_stmt *stmt;
				int sample_id;

				std::string query = ""
					"SELECT sample_id "
					"FROM samples_matches "
					"WHERE leaf_id=?";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[get_samples|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				// set values
				rc = sqlite3_bind_int(stmt, 1, leaf_id);
				if(rc) {
					std::stringstream msg("");
					msg << "[get_samples|2] Cannot bind value [leaf_id] to statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				while(true){
					rc = sqlite3_step(stmt);
					if(rc == SQLITE_DONE){
						break;
					}else if(rc == SQLITE_ROW){
						sample_id = sqlite3_column_int(stmt, 0);
						result.push_back(sample_id);
					}
				}

				sqlite3_finalize(stmt);
				return(result);
			}

			std::vector<std::vector<int>> get_samples_id_by_leaf(){
				std::vector<std::vector<int>> result;
				std::vector<int> by_leaf;
				sqlite3_stmt *stmt;
				int leaf_id=0, aux, sample_id, i;

				std::string query = ""
					"SELECT leaf_id, sample_id "
					"FROM samples_matches "
					"ORDER BY leaf_id ASC";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[get_samples_id_by_leaf|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				while(true){
					rc = sqlite3_step(stmt);
					if(rc == SQLITE_DONE){
						result.push_back(by_leaf);
						break;
					}else if(rc == SQLITE_ROW){
						aux = sqlite3_column_int(stmt, 0);
						sample_id = sqlite3_column_int(stmt, 1);
						if(aux==leaf_id){
							by_leaf.push_back(sample_id);
						}else if(aux>leaf_id){
							result.push_back(by_leaf);
							for(i=leaf_id+1; i<aux; i++){
								result.push_back({});
							}
							leaf_id = aux;
							by_leaf = {sample_id};
						}else{
							std::stringstream msg("");
							msg << "[get_samples_id_by_leaf|1] Error retrieving the data";
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
					}
				}

				sqlite3_finalize(stmt);
				return(result);
			}

			std::vector<std::vector<int>> get_leaves_id_by_tree(){
				std::vector<std::vector<int>> result;
				std::vector<int> by_leaf;
				sqlite3_stmt *stmt;
				int tree_id=0, aux, leaf_id, i;

				std::string query = ""
					"SELECT DISTINCT tree_id, leaf_id "
					"FROM samples_matches "
					"ORDER BY tree_id ASC";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[get_leaves_id_by_tree|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				while(true){
					rc = sqlite3_step(stmt);
					if(rc == SQLITE_DONE){
						result.push_back(by_leaf);
						break;
					}else if(rc == SQLITE_ROW){
						aux = sqlite3_column_int(stmt, 0);
						leaf_id = sqlite3_column_int(stmt, 1);
						if(aux==tree_id){
							by_leaf.push_back(leaf_id);
						}else if(aux>tree_id){
							result.push_back(by_leaf);
							for(i=tree_id+1; i<aux; i++){
								result.push_back({});
							}
							tree_id = aux;
							by_leaf = {leaf_id};
						}else{
							std::stringstream msg("");
							msg << "[get_leaves_id_by_tree|1] Error retrieving the data";
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
					}
				}

				sqlite3_finalize(stmt);
				return(result);
			}

			std::vector<std::vector<float>> load_coords(){
				std::vector<std::vector<float>> result;
				std::vector<float> _coords;
				sqlite3_stmt *stmt;
				int sample_id=0, n, aux;

				std::string query = ""
					"SELECT sample_id, axis, coord "
					"FROM positions "
					"ORDER BY sample_id, axis;";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[load_coords|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				while(true){
					rc = sqlite3_step(stmt);
					if(rc == SQLITE_DONE){
						if(0<_coords.size()){
							result.push_back(_coords);
						}
						break;
					}else if(rc == SQLITE_ROW){
						aux = sqlite3_column_int(stmt, 0);
						if(aux!=sample_id){ // new sample_id
							// store last
							sample_id = aux;
							result.push_back(_coords);
							aux = sqlite3_column_int(stmt, 1);
							if(0 != aux){
								sqlite3_close_v2(this->db);
								throw std::runtime_error("[load_coords|2] axis 0 missing for sample_id '" + std::to_string(aux) + "'");
							}
							_coords = {(float)sqlite3_column_double(stmt, 2)};
						}else{
							aux = sqlite3_column_int(stmt, 1);
							n = _coords.size();
							if(n != aux){
								sqlite3_close_v2(this->db);
								throw std::runtime_error("[load_coords|3] axis " + std::to_string(_coords.size()) + " missing for sample_id '" + std::to_string(aux) + "'");
							}
							_coords.push_back((float)sqlite3_column_double(stmt, 2));
						}
					}else{
						std::stringstream msg("");
						msg << "[load_coords|4] Cannot retrieve data: ";
						msg << sqlite3_errmsg(db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
				}

				sqlite3_finalize(stmt);
				return(result);
			}

			std::vector<float> load_values(){
				std::vector<float> result;
				sqlite3_stmt *stmt;

				std::string query = ""
					"SELECT id, value "
					"FROM samples "
					"ORDER BY samples.id;";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[load_values|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				while(true){
					rc = sqlite3_step(stmt);
					if(rc == SQLITE_DONE){
						break;
					}else if(rc == SQLITE_ROW){
						result.push_back(sqlite3_column_double(stmt, 1));
					}else{
						std::stringstream msg("");
						msg << "[load_values|4] Cannot retrieve data: ";
						msg << sqlite3_errmsg(db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
				}

				sqlite3_finalize(stmt);
				return(result);
			}

			std::vector<std::vector<float>> load_extras(){
				std::vector<std::vector<float>> result;
				std::vector<float> _values;
				sqlite3_stmt *stmt;
				int leaf_id=-1, n, aux;

				std::string query = ""
					"SELECT leaf_id, elem, value "
					"FROM extras "
					"ORDER BY leaf_id, elem;";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[load_extras|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				while(true){
					rc = sqlite3_step(stmt);
					if(rc == SQLITE_DONE){
						if(0<_values.size()){
							result.push_back(_values);
						}
						break;
					}else if(rc == SQLITE_ROW){
						aux = sqlite3_column_int(stmt, 0);
						if(aux!=leaf_id){ // new leaf_id
							// store last
							leaf_id = aux;
							result.push_back(_values);
							aux = sqlite3_column_int(stmt, 1);
							if(0 != aux){
								sqlite3_close_v2(this->db);
								throw std::runtime_error("[load_extras|2] elem 0 missing for leaf_id '" + std::to_string(aux) + "'");
							}
							_values = {(float)sqlite3_column_double(stmt, 2)};
						}else{
							aux = sqlite3_column_int(stmt, 1);
							n = _values.size();
							if(n != aux){
								sqlite3_close_v2(this->db);
								throw std::runtime_error("[load_extras|3] elem " + std::to_string(_values.size()) + " missing for leaf_id '" + std::to_string(aux) + "'");
							}
							_values.push_back((float)sqlite3_column_double(stmt, 2));
						}
					}else{
						std::stringstream msg("");
						msg << "[load_extras|4] Cannot retrieve data: ";
						msg << sqlite3_errmsg(db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}
				}

				sqlite3_finalize(stmt);
				return(result);
			}

			void create_tree(int *id_count, int tree_id, float lambda, std::vector<float> bbox, int seed=28){
				int ndim = (int) bbox.size()/2;
				std::mt19937 my_rand(seed);
				std::uniform_int_distribution<int> uni_int(0, ndim-1);
				std::uniform_real_distribution<float> uni_float(0, 1);
				std::exponential_distribution<float> exp_float;
				std::queue<std::vector<float>> leaves_queue, bbox_queue;

				std::vector<float> aux_node, cur_node = {(float)((*id_count)++), 0.0, -1.0, 0.0, -1.0, -1.0, -1.0}; // id, tau, cut, height, axis, lower, greater
				std::vector<float> aux_bbox, cur_bbox;
				float aux_value;

				if(lambda > 0){
					// add first leaf to the queue
					leaves_queue.push(cur_node);
					bbox_queue.push(bbox);
					this->add_root(tree_id, cur_node.at(0));
				}

				// build the tree[
				while(!leaves_queue.empty()){
					// get first in the queue
					cur_node = leaves_queue.front();
					leaves_queue.pop();

					if (cur_node.at(1) < lambda){
						// bounding box for cur_node
						cur_bbox = bbox_queue.front();
						bbox_queue.pop();

						// component to split
						cur_node.at(4) = uni_int(my_rand);
						// cut in min + (max-min)*random(0,1)
						cur_node.at(2) = cur_bbox.at(2*cur_node.at(4)) + (cur_bbox.at(2*cur_node.at(4)+1) - cur_bbox.at(2*cur_node.at(4))) * uni_float(my_rand);

						// LOWER THAN CUT
						// copy bbox
						aux_bbox = cur_bbox;
						// modify upper bound of split axis
						aux_bbox.at(2*cur_node.at(4)+1) = cur_node.at(2);
						// sum of bbox ranges
						aux_value = bbox_sum_interval(aux_bbox);
						// set values for exponential distribution
						exp_float.param(std::exponential_distribution<float>::param_type(aux_value));
						// get the tau for the lower child
						aux_value = cur_node.at(1) + exp_float(my_rand);
						// the new child
						aux_node = {(float)((*id_count)++), aux_value, -1.0, cur_node.at(3)+1, -1.0, -1.0, -1.0};
						// add new node reference in cur node (parent)
						cur_node.at(5) = aux_node.at(0);
						// verify if new node must be splitted
						if(aux_value<lambda){
							// added to the queue to be split
							leaves_queue.push(aux_node);
							bbox_queue.push(aux_bbox);
						}else{
							// added to the db
							this->add_leaf(tree_id, &aux_node);
							this->add_bbox(tree_id, (int) aux_node.at(0), &aux_bbox);
						}

						// GREATER THAN CUT
						// copy bbox
						aux_bbox = cur_bbox;
						// modify lower bound of split axis
						aux_bbox.at(2*cur_node.at(4)) = cur_node.at(2);
						// sum of bbox ranges
						aux_value = bbox_sum_interval(aux_bbox);
						// set values for exponential distribution
						exp_float.param(std::exponential_distribution<float>::param_type(aux_value));
						// get the tau for the lower child
						aux_value = cur_node.at(1) + exp_float(my_rand);
						// the new child
						aux_node = {(float)((*id_count)++), aux_value, -1.0, cur_node.at(3)+1, -1.0, -1.0, -1.0};
						// add new node reference in cur node (parent)
						cur_node.at(6) = aux_node.at(0);
						// verify if new node must be splitted
						if(aux_value<lambda){
							// added to the queue to be split
							leaves_queue.push(aux_node);
							bbox_queue.push(aux_bbox);
						}else{
							// added to the db
							this->add_leaf(tree_id, &aux_node);
							this->add_bbox(tree_id, (int) aux_node.at(0), &aux_bbox);
						}

						// add current node to the db
						this->add_leaf(tree_id, &cur_node);
						this->add_bbox(tree_id, (int) cur_node.at(0), &cur_bbox);
					}
				}
			}

			bool exist_param(std::string key){
				bool result = false;
				sqlite3_stmt *stmt;
				std::string query = "SELECT value FROM params WHERE key='" + key + "';";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[get_param_as_int] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				// get value
				rc = sqlite3_step(stmt);
				if(rc == SQLITE_ROW){
					result = true;
				}

				sqlite3_finalize(stmt);
				return(result);
			}

			void set_param(std::string key, int value){
				sqlite3_stmt *stmt;
				int rc;
				std::string query;

				if(this->exist_param(key)){
					// modify

					query = "UPDATE 'params' SET type='int' value='" + std::to_string(value) + "' WHERE key='" + key + "';";
					rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

					// compilation not ok
					if(rc) {
						std::stringstream msg("");
						msg << "[set_param|1] Cannot compile statememt: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}

					this->begin_transaction();
						rc = sqlite3_step(stmt);
						if(rc != SQLITE_DONE){
							std::stringstream msg("");
							msg << "[set_param|2] Modification fails: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
					this-> end_transaction();
				}else{
					// insertion
					query = "INSERT INTO 'params' (key, type, value) VALUES('" + key + "','int','" + std::to_string(value) + "');";
					rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

					// compilation not ok
					if(rc) {
						std::stringstream msg("");
						msg << "[set_param|3] Cannot compile statememt: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}

					this->begin_transaction();
						rc = sqlite3_step(stmt);
						if(rc != SQLITE_DONE){
							std::stringstream msg("");
							msg << "[set_param|4] Insertion fails: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
					this-> end_transaction();
				}

				sqlite3_finalize(stmt);
			}

			void set_param(std::string key, float value){
				sqlite3_stmt *stmt;
				int rc;
				std::string query;

				if(this->exist_param(key)){
					// modify
					query = "UPDATE 'params' SET type='float' value='" + std::to_string(value) + "' WHERE key='" + key + "';";
					rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

					// compilation not ok
					if(rc) {
						std::stringstream msg("");
						msg << "[set_param|5] Cannot compile statememt: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}

					this->begin_transaction();
						rc = sqlite3_step(stmt);
						if(rc != SQLITE_DONE){
							std::stringstream msg("");
							msg << "[set_param|6] Modification fails: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
					this-> end_transaction();
				}else{
					// insertion
					query = "INSERT INTO 'params' (key, type, value) VALUES('" + key + "','float','" + std::to_string(value) + "');";
					rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

					// compilation not ok
					if(rc) {
						std::stringstream msg("");
						msg << "[set_param|7] Cannot compile statememt: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}

					this->begin_transaction();
						rc = sqlite3_step(stmt);
						if(rc != SQLITE_DONE){
							std::stringstream msg("");
							msg << "[set_param|8] Insertion fails: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
					this-> end_transaction();
				}

				sqlite3_finalize(stmt);
			}

			void set_param(std::string key, std::string value){
				sqlite3_stmt *stmt;
				int rc;
				std::string query;

				if(this->exist_param(key)){
					// modify
					query = "UPDATE 'params' SET type='text' value='" + value + "' WHERE key='" + key + "';";
					rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

					// compilation not ok
					if(rc) {
						std::stringstream msg("");
						msg << "[set_param|9] Cannot compile statememt: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}

					this->begin_transaction();
						rc = sqlite3_step(stmt);
						if(rc != SQLITE_DONE){
							std::stringstream msg("");
							msg << "[set_param|10] Modification fails: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
					this-> end_transaction();
				}else{
					// insertion
					query = "INSERT INTO 'params' (key, type, value) VALUES('" + key + "','text','" + value + "');";
					rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

					// compilation not ok
					if(rc) {
						std::stringstream msg("");
						msg << "[set_param|11] Cannot compile statememt: ";
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}

					this->begin_transaction();
						rc = sqlite3_step(stmt);
						if(rc != SQLITE_DONE){
							std::stringstream msg("");
							msg << "[set_param|12] Insertion fails: ";
							msg << sqlite3_errmsg(this->db);
							sqlite3_close_v2(this->db);
							throw std::runtime_error(msg.str());
						}
					this-> end_transaction();
				}

				sqlite3_finalize(stmt);
			}

			std::string get_text_param(std::string key){
				std::string result = "";
				sqlite3_stmt *stmt;
				std::string query = "SELECT type, value FROM params WHERE key='" + key + "';";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[get_text_param|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				// get value
				rc = sqlite3_step(stmt);
				if(rc == SQLITE_ROW){
					result = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
					if(result!="text"){
						std::stringstream msg("");
						msg << "[get_text_param|2] The value for key '" << key << "' was stored as " << result;
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}else{
						result = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
					}
				}

				sqlite3_finalize(stmt);
				return(result);
			}

			float get_float_param(std::string key){
				std::string result = "";
				sqlite3_stmt *stmt;
				std::string query = "SELECT type, value FROM params WHERE key='" + key + "';";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[get_float_param|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				// get value
				rc = sqlite3_step(stmt);
				if(rc == SQLITE_ROW){
					result = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
					if((result!="float") && (result!="int")){
						std::stringstream msg("");
						msg << "[get_float_param|3] The value for key '" << key << "' was stored as " << result;
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}else{
						result = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
					}
				}

				sqlite3_finalize(stmt);
				return(std::stof(result));
			}

			int get_int_param(std::string key){
				std::string result = "";
				sqlite3_stmt *stmt;
				std::string query = "SELECT type, value FROM params WHERE key='" + key + "';";
				int rc = sqlite3_prepare_v3(this->db, query.c_str(), query.length(), 0, &stmt, NULL);

				// compilation not ok
				if(rc) {
					std::stringstream msg("");
					msg << "[get_int_param|1] Cannot compile statememt: ";
					msg << sqlite3_errmsg(this->db);
					sqlite3_close_v2(this->db);
					throw std::runtime_error(msg.str());
				}

				// get value
				rc = sqlite3_step(stmt);
				if(rc == SQLITE_ROW){
					result = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
					if(result!="int"){
						std::stringstream msg("");
						msg << "[get_int_param|3] The value for key '" << key << "' was stored as " << result;
						msg << sqlite3_errmsg(this->db);
						sqlite3_close_v2(this->db);
						throw std::runtime_error(msg.str());
					}else{
						result = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
					}
				}

				sqlite3_finalize(stmt);
				return(std::stoi(result));
			}

			virtual std::vector<float> *get_extra(int leaf_id){
				return(&(this->extras.at(leaf_id)));
			}

			// Override
			virtual std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *queries, std::vector<int> *queries_id, std::vector<float> *extra){
				throw std::runtime_error("[leaf_estimation] must override");
			}

			virtual std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *extra){
				throw std::runtime_error("[leaf_loo] must override");
			}

			virtual std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<int> *fold, std::vector<float> *extra){
				throw std::runtime_error("[leaf_kfold] must override");
			}

			virtual std::vector<std::vector<float>> post_process(){
				throw std::runtime_error("[post_process] must override");
			}

		public:

			ESI(std::string path, std::vector<std::vector<float>> *_coords, std::vector<float> *_values, int n_tree, float alpha, std::vector<float> bbox, int seed=2007203, int random_range=206936){
				int i, n;
				std::mt19937 my_rand(seed);
				std::uniform_int_distribution<int> uni_int(0, random_range);

                std::cout << "[spatialite] opening database ..." << std::endl;
				// open the database
				this->open_database(path);

                std::cout << "[spatialite] creating tables ..." << std::endl;
				// create tree tables
				this->create_table("leaves", "CREATE TABLE 'leaves'('id' INTEGER PRIMARY KEY, 'tree_id' INTEGER, 'tau' REAL, 'cut' REAL, 'height' INTEGER, 'axis' INTEGER, 'lower' INTEGER REFERENCES leaves, 'greater' INTEGER REFERENCES leaves);");
				this->create_table("bboxes", "CREATE TABLE 'bboxes'('id' INTEGER PRIMARY KEY, 'tree_id' INTEGER, 'leaf_id' INTEGER REFERENCES leaves, 'axis' INTEGER, 'lower_bound' REAL, 'upper_bound' REAL);");
				this->create_table("roots", "CREATE TABLE 'roots'('id' INTEGER PRIMARY KEY, 'leaf_id' INTEGER);");
				// create points tables
				this->create_table("positions", "CREATE TABLE 'positions'('id' INTEGER PRIMARY KEY, 'sample_id' INTEGER, 'axis' INTEGER, 'coord' REAL);");
				this->create_table("samples", "CREATE TABLE 'samples'('id' INTEGER PRIMARY KEY, 'value' REAL);");
				this->create_table("queries", "CREATE TABLE 'queries'('id' INTEGER PRIMARY KEY, 'query_id' INTEGER, 'axis' INTEGER, 'coord' REAL);");
				// create aux tables
				this->create_table("samples_matches", "CREATE TABLE 'samples_matches'('id' INTEGER PRIMARY KEY AUTOINCREMENT, 'tree_id' INTEGER, 'sample_id' INTEGER, 'leaf_id' INTEGER);");
				this->create_table("queries_matches", "CREATE TABLE 'queries_matches'('id' INTEGER PRIMARY KEY AUTOINCREMENT, 'tree_id' INTEGER, 'query_id' INTEGER, 'leaf_id' INTEGER);");
				this->create_table("params", "CREATE TABLE 'params'('key' TEXT PRIMARY KEY, 'type' TEXT, 'value' TEXT);");
				this->create_table("extras", "CREATE TABLE 'extras'('id' INTEGER PRIMARY KEY AUTOINCREMENT, 'leaf_id' INTEGER, 'elem' INTEGER, 'value' REAL);");

				this->coords = *(_coords);
				this->values = *(_values);

				int n_coords = this->coords.size(), n_dims = this->coords.at(0).size();
				this->set_param("n_tree", n_tree);
				this->set_param("n_coords", n_coords);
				this->set_param("n_dims", n_dims);

				// check values and coords have same length
				n = this->values.size();
				if(n != n_coords){
					throw std::runtime_error("[ESI|1] coords and values have different length");
				}

				// check all samples have n_dims components
				for(i=0; i<n_coords; i++){
					n = this->coords.at(i).size();
					if(n != n_dims){
						throw std::runtime_error("[ESI|2] Not all coords have same dimensions (check positions 0 and " + std::to_string(i) + ")");
					}
				}

                std::cout << "[spatialite] adding samples ..." << std::endl;
				// add samples to db
				this->add_samples(&(this->coords), &(this->values));

				// trees params
				int id_count = 0;
				float lambda = bbox_sum_interval(bbox);
				lambda = 1.0/(lambda - alpha*lambda);

                std::cout << "[spatialite] building trees (" << n_tree << ") ..." << std::endl;
				// build trees
				for(i=0; i<n_tree; i++){
				    std::cout << "[spatialite] current tree: " << i << " ..." << std::endl;
					this->create_tree(&id_count, i, lambda, bbox, uni_int(my_rand));
				}

                std::cout << "[spatialite] setting samples to leaves ..." << std::endl;
				// set samples to leaves
				this->set_samples_to_leaves();
				std::cout << "[spatialite] done." << std::endl;
			}

			ESI(std::string path){
				// open the database
				this->open_database(path);

				// retrieve coords and values
				this->coords = this->load_coords();
				this->values = this->load_values();
			}

			std::vector<std::vector<float>> estimate(std::vector<std::vector<float>> *locations, std::function<int(std::string)> visitor){
				std::stringstream json;
				std::vector<std::pair<int,std::vector<int>>> leaf_queries;
				std::vector<std::vector<float>> result;
				std::vector<std::vector<int>> leaf_samples;
				std::vector<float> *extra, r;
				std::vector<int> samples_id;

				int i_loc, i_tree, i_query, n;
				int n_loc = locations->size();
				int n_dims = this->get_int_param("n_dims");
				int n_tree = this->get_int_param("n_tree");

				// check locations have same dimensions as samples
				for(i_loc=0; i_loc<n_loc; i_loc++){
					n = locations->at(i_loc).size();
					if(n != n_dims){
						throw std::runtime_error("[estimation|1] locations do no have same dimensions as samples (check position" + std::to_string(i_loc) + ")");
					}
				}

				// clear the queries table
				this->clear_queries();
				// add queries to db
				this->add_queries(locations);
				// get leaf for every location in every tree
				this->set_queries_to_leaves();
				// retrive all samples by the leaf where they belong
				leaf_samples = this->get_samples_id_by_leaf();
				// create result array
				n = locations->size();
				for(i_query=0; i_query<n; i_query++){
					result.push_back({});
				}

				// for all locations
				for(i_tree=0; i_tree<n_tree; i_tree++){
					leaf_queries = this->get_leaves_n_queries(i_tree);
					for(auto l_q : leaf_queries){
						extra = this->get_extra(l_q.first);
						samples_id = leaf_samples.at(l_q.first);
						r = this->leaf_estimation(&(this->coords), &(this->values), &samples_id, locations, &(l_q.second), extra);
						n = l_q.second.size();
						for(i_query=0; i_query<n; i_query++){
							result.at(l_q.second.at(i_query)).push_back(r.at(i_query));
						}
					}
					// visitor execution
					json.str("");
					json << "{\"percentage\": " << 100.0*(i_tree+1.0)/n_tree << "}";
					visitor(json.str());
				}

				return(result);
			}

			std::vector<std::vector<float>> estimate(std::vector<std::vector<float>> *locations){
				return(this->estimate(locations, [](std::string s){return(0);}));
			}

			std::vector<std::vector<float>> leave_one_out(std::function<int(std::string)> visitor){
				std::stringstream json;
				std::vector<std::vector<int>> s_by_l, l_by_t;
				std::vector<std::vector<float>> result;
				std::vector<float> *extra, r;
				std::vector<int> leaves_id, samples_id;
				int i_tree, i_leaf, i_smp, n_tree = this->get_int_param("n_tree"), n_leaves, n_smps;

				s_by_l = this->get_samples_id_by_leaf();
				l_by_t = this->get_leaves_id_by_tree();

				n_smps = this->coords.size();
				for(i_smp=0; i_smp<n_smps; i_smp++){
					result.push_back({});
				}

				for(i_tree=0; i_tree<n_tree; i_tree++){
					leaves_id = l_by_t.at(i_tree);
					n_leaves = leaves_id.size();
					for(i_leaf=0; i_leaf<n_leaves; i_leaf++){
						extra = this->get_extra(leaves_id.at(i_leaf));
						samples_id = s_by_l.at(leaves_id.at(i_leaf));
						r = leaf_loo(&(this->coords), &(this->values), &samples_id, extra);
						n_smps = samples_id.size();
						for(i_smp=0; i_smp<n_smps; i_smp++){
							result.at(samples_id.at(i_smp)).push_back(r.at(i_smp));
						}
					}
					// visitor execution
					json.str("");
					json << "{\"percentage\": " << 100.0*(i_tree+1.0)/n_tree << "}";
					visitor(json.str());
				}

				return(result);
			}

			std::vector<std::vector<float>> leave_one_out(){
				return(this->leave_one_out([](std::string s){return(0);}));
			}

			std::vector<std::vector<float>> k_fold(int k, std::function<int(std::string)> visitor, int seed=2007203){
				std::mt19937 my_rand(seed);
				std::uniform_real_distribution<float> uni_float;
				auto folds = get_folds(values.size(), k, uni_float(my_rand));

				std::stringstream json;
				std::vector<std::vector<int>> s_by_l, l_by_t;
				std::vector<std::vector<float>> result;
				std::vector<float> *extra, r;
				std::vector<int> leaves_id, samples_id;
				int i_tree, i_leaf, i_smp, n_tree = this->get_int_param("n_tree"), n_leaves, n_smps;

				s_by_l = this->get_samples_id_by_leaf();
				l_by_t = this->get_leaves_id_by_tree();

				n_smps = this->coords.size();
				for(i_smp=0; i_smp<n_smps; i_smp++){
					result.push_back({});
				}

				for(i_tree=0; i_tree<n_tree; i_tree++){
					leaves_id = l_by_t.at(i_tree);
					n_leaves = leaves_id.size();
					for(i_leaf=0; i_leaf<n_leaves; i_leaf++){
						extra = this->get_extra(leaves_id.at(i_leaf));
						samples_id = s_by_l.at(leaves_id.at(i_leaf));
						r = leaf_kfold(k, &(this->coords), &(this->values), &folds, &samples_id, extra);
						n_smps = samples_id.size();
						for(i_smp=0; i_smp<n_smps; i_smp++){
							result.at(samples_id.at(i_smp)).push_back(r.at(i_smp));
						}
					}
					// visitor execution
					json.str("");
					json << "{\"percentage\": " << 100.0*(i_tree+1.0)/n_tree << "}";
					visitor(json.str());
				}

				return(result);


				// std::stringstream json;
				// auto fold_rand = std::mt19937(seed);
				// std::uniform_real_distribution<float> uni_float;
				// auto folds = get_folds(values.size(), k, uni_float(fold_rand));
				// std::vector<std::vector<float>> results(coords.size());
				// int n = mondrian_forest.size();
				//
				// for(int i=0; i<n; i++){
				// 	// get tree
				// 	if(this->debug){std::cout << "i : " << i << std::endl;}
				// 	auto mt = mondrian_forest.at(i);
				// 	// make kfold by leaf
				// 	for(int j=0; j<mt->samples_by_leaf.size(); j++){
				// 		if(this->debug){std::cout << "  j : " << j << std::endl;}
				// 		if(mt->samples_by_leaf.at(j).size()!=0){
				// 			for(int k=0; k<mt->samples_by_leaf.at(j).size(); k++){
				// 				results.at(mt->samples_by_leaf.at(j).at(k)).push_back(predictions.at(k));
				// 			}
				// 		}
				// 	}
				// 	json.str("");
				// 	json << "{\"percentage\": " << 100.0*(i+1.0)/n << "}";
				// 	visitor(json.str());
				// }
				// return(results);
			}

			std::vector<std::vector<float>> k_fold(int k, int seed=2007203){
				return(this->k_fold(k, [](std::string s){return(0);}, seed));
			}
	};
}

#endif
