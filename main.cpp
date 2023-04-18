#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <ctime>
#include <functional>
#include "spatialize/file.hpp"
#include "spatialize/utils.hpp"
#include "spatialize/esi_idw.hpp"
#include "spatialize/esi_kriging.hpp"
#include "spatialize/agg.hpp"

void test(){
  std::chrono::duration<float> creation, estimation, aggregation;
    auto coords = sptlz::readSamplesFromGSLIB<float>("/Users/fnv/Projects/spatialize/build/test/tesdata/samples.dat", {"x", "y"});
    auto values_1xn = sptlz::readSamplesFromGSLIB<float>("/Users/fnv/Projects/spatialize/build/test/tesdata/samples.dat", {"cu"});
    // auto values_1xn = sptlz::readSamplesFromCSV<float>("/home/fgarrido/Dropbox/alges-dev/justesi/data/data.csv", {"cu"});
    auto values = sptlz::as_1d_array(&values_1xn);
    auto locations = sptlz::readSamplesFromCSV<float>("/Users/fnv/Projects/spatialize/build/test/tesdata/grid.csv", {"x", "y"});

    std::cout << "READ OK" << std::endl;
    auto bbox = sptlz::samples_coords_bbox(locations);
    float lambda = sptlz::bbox_sum_interval(bbox);
    float alpha = 0.7;
    float forest_size = 100;
    lambda = 1/(lambda-alpha*lambda);
    std::cout << "Starting" << std::endl;
  auto ti = std::chrono::system_clock::now();
    sptlz::ESI_IDW* esi = new sptlz::ESI_IDW(coords, values, lambda, forest_size, bbox, 2.0, 15933231);
    // sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(coords, values, lambda, forest_size, bbox, 1, 0.1, 5000.0, 15933231);
  creation = std::chrono::system_clock::now()-ti;
    std::cout << "ESI_IDW created" << std::endl;
    // std::cout << "ESI_Kriging created" << std::endl;
  ti = std::chrono::system_clock::now();
    auto r = esi->estimate(&locations);
  estimation = std::chrono::system_clock::now()-ti;
    std::cout << "estimation performed" << std::endl;
  ti = std::chrono::system_clock::now();
    auto output = sptlz::average(r);
  aggregation = std::chrono::system_clock::now()-ti;
    std::cout << "Aggregation made" << std::endl;

    std::ofstream fout;
    fout.open("/Users/fnv/Projects/spatialize/build/test/tesdata/esi_idw_e2.csv");
    // fout.open("/home/fgarrido/esi_idw_anis.csv");
    // fout.open("/home/fgarrido/esi_kri.csv");
//    fout << "X,Y,V";
//    for(int i=0; i<forest_size; i++){
//      fout << ",V" << i;
//    }
//    fout << std::endl;
//    for(int i=0; i<output.size(); i++){
//      fout << locations.at(i).at(0) << "," << locations.at(i).at(1) << "," << output.at(i);
//      for(int j=0; j<forest_size; j++){
//        fout << "," << r.at(i).at(j);
//      }
//      fout << std::endl;
//    }
    fout << "X,Y,V" << std::endl;
    for(int i=0; i<output.size(); i++){
      fout << locations.at(i).at(0) << "," << locations.at(i).at(1) << "," << output.at(i) << std::endl;
    }
    fout.close();
    std::cout << "File saved" << std::endl;
  std::cout << "Creation    : " << creation.count() << " [s]" << std::endl;
  std::cout << "Estimation  : " << estimation.count() << " [s]" << std::endl;
  std::cout << "Aggregation : " << aggregation.count() << " [s]" << std::endl;
}

int main(int argc, char* argv[]){
  test();
}
