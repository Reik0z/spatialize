#ifndef _SPTLZ_STRING_
#define _SPTLZ_STRING_

#include <vector>
#include <string>
#include <algorithm>

namespace sptlz{
  std::string ltrim(std::string &txt){
    int i;
    for(i=0;i<txt.length();i++){
      if(!std::isspace(txt[i])){
        break;
      }
    }

    return(txt.substr(i));
  }

  std::string rtrim(std::string &txt){
    int i;
    for(i=txt.length()-1;i>=0;i--){
      if(!std::isspace(txt[i])){
        break;
      }
    }

    return(txt.substr(0,i+1));
  }

  std::string trim(std::string &txt){
    int i,j;
    for(i=0;i<txt.length();i++){
      if(!std::isspace(txt[i])){
        break;
      }
    }
    for(j=txt.length()-1;j>=0;j--){
      if(!std::isspace(txt[j])){
        break;
      }
    }

    return(txt.substr(i,j-i+1));
  }

  std::vector<std::string> split(const std::string& txt){
    std::vector<std::string> pieces, result;
    std::string bitmap;

    std::transform(txt.begin(), txt.end() , std::back_inserter(bitmap), [](const char& c){
      if(std::isspace(c)){
        return('1');
      }else{
        return('0');
      }
    });

    int prev = 0, current;
    std::string piece;

    while((current = bitmap.find("1", prev)) != std::string::npos){
      pieces.push_back(txt.substr(prev, current-prev));
      prev = current + 1;
    }
    pieces.push_back(txt.substr(prev));

    std::copy_if(pieces.begin(), pieces.end() , std::back_inserter(result), [](const std::string& s){return(s.length()>0);});

    return(result);
  }

  std::vector<std::string> split(const std::string& txt, const std::string& sep){
    std::vector<std::string> pieces;
    int prev = 0, current, l_sep = sep.length();
    std::string piece;

    while((current = txt.find(sep, prev)) != std::string::npos){
      pieces.push_back(txt.substr(prev, current-prev));
      prev = current + l_sep;
    }
    pieces.push_back(txt.substr(prev));

    return(pieces);
  }

  std::string join(const std::vector<std::string> list, const std::string& sep){
    std::string result;
    for(int i=0;i<list.size();i++){
      if (i>0){
        result += sep;
      }
      result += list[i];
    }
    return(result);
  }
}

#endif
