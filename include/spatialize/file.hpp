#ifndef _SPTLZ_FILE_
#define _SPTLZ_FILE_

#include <fstream>
#include <vector>
#include <string>
#include "spatialize/string.hpp"

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
}

#endif
