#ifndef _SPTLZ_FILE_
#define _SPTLZ_FILE_

#include <fstream>
#include <vector>
#include <string>
#include "string.hpp"

namespace sptlz{
  std::vector<std::string> readFile(const std::string& path){
    std::ifstream the_file(path);
    std::string the_line;
    std::vector<std::string> lines = {};

    while(getline(the_file, the_line)) {
      lines.push_back(the_line);
    }
    the_file.close();

    return(lines);
  }

  std::vector<std::vector<std::string>> readCSV(const std::string& path){
    std::vector<std::vector<std::string>> records = {};

    auto lines = readFile(path);
    if(lines.size()==0){
      return(records);
    }

    for(auto &line : lines){
      records.push_back(sptlz::split(line, ","));
    }

    return(records);
  }

  std::vector<std::vector<std::string>> readGSLIB(const std::string& path){
    std::vector<std::vector<std::string>> records = {};

    auto lines = readFile(path);
    if(lines.size()==0){
      return(records);
    }
    auto n_vars = atoi(lines[1].c_str());
    std::vector<std::string> header;
    for(int i=0;i<n_vars;i++){
      header.push_back(sptlz::trim(lines[2+i]));
    }

    records.push_back(header);

    for(auto it=lines.begin()+2+n_vars;it!=lines.end();++it){
      records.push_back(sptlz::split(*it));
    }

    return(records);
  }

  template <class T>
  std::vector<std::vector<T>> readSamplesFromCSV(const std::string& path, std::vector<std::string> variables){
    std::vector<std::vector<T>> samples = {};
    std::vector<int> idxs = {};
    std::vector<T> current;
    T number;

    auto records = readCSV(path);
    if(records.size()==0){
      return(samples);
    }
    for(int i=0;i<variables.size();i++){
      for(int j=0;j<records[0].size();j++){
        if(variables[i]==records[0][j]){
          idxs.push_back(j);
        }
      }
    }

    for(auto it=std::next(records.begin());it!=records.end();++it){
      current = {};
      for(auto idx : idxs){
        number = std::stold(it->at(idx).c_str());
        current.push_back((T)number);
      }
      samples.push_back(current);
    }

    return samples;
  }

  template <class T>
  std::vector<std::vector<T>> readSamplesFromGSLIB(const std::string& path, std::vector<std::string> variables){
    std::vector<std::vector<T>> samples = {};
    std::vector<int> idxs = {};
    std::vector<T> current;
    T number;

    auto records = readGSLIB(path);
    for(int i=0;i<variables.size();i++){
      for(int j=0;j<records[0].size();j++){
        if(variables[i]==records[0][j]){
          idxs.push_back(j);
        }
      }
    }

    for(auto it=std::next(records.begin());it!=records.end();++it){
      current = {};
      for(auto idx : idxs){
        number = std::stold(it->at(idx).c_str());
        current.push_back((T)number);
      }
      samples.push_back(current);
    }

   return samples;
  }

  template <char sep>
  void writeCSV(const std::string& path, std::vector<std::string> header, std::vector<std::vector<float>> *values){
    // TODO check square
    int n_line=values->size(), n_cols=values->at(0).size();
    
    std::ofstream the_file(path);
    int i,j;

    // header
    the_file << header.at(0);
    for(i=1; i<n_cols; i++){
      the_file << sep << header.at(i);
    }
    the_file << std::endl;

    // values
    for(i=0; i<n_line; i++){
      the_file << values->at(i).at(0);
      for(j=1; j<n_cols; j++){
        the_file << sep << values->at(i).at(j);
      }
      the_file << std::endl;
    }

    the_file.close();
  }

  template <char sep>
  void writeCSV(const std::string& path, std::vector<std::string> header1, std::vector<std::string> header2, std::vector<std::vector<float>> *values1, std::vector<std::vector<float>> *values2){
    // TODO check square and same n_line
    int n_line=values1->size(), n_cols1=values1->at(0).size(), n_cols2=values2->at(0).size();
    
    std::ofstream the_file(path);
    int i,j;

    // header
    the_file << header1.at(0);
    for(i=1; i<n_cols1; i++){
      the_file << sep << header1.at(i);
    }
    for(i=0; i<n_cols2; i++){
      the_file << sep << header2.at(i);
    }
    the_file << std::endl;

    // values
    for(i=0; i<n_line; i++){
      the_file << values1->at(i).at(0);
      for(j=1; j<n_cols1; j++){
        the_file << sep << values1->at(i).at(j);
      }
      for(j=0; j<n_cols2; j++){
        the_file << sep << values2->at(i).at(j);
      }
      the_file << std::endl;
    }

    the_file.close();
  }
}

#endif
