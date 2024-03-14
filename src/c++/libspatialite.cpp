#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <vector>
#include <ctime>
#include "spatialite/abstract_esi.hpp"
#include "spatialite/esi_idw.hpp"
#include "spatialite/esi_kriging.hpp"
#include "spatialite/utils.hpp"

sptlz::ESI *get_esi_model(std::string path){
  std::string type = sptlz::get_esi_type(path);
  if(type == "idw"){
    return(new sptlz::ESI_IDW(path));
  }else if(type == "kriging"){
    return(new sptlz::ESI_Kriging(path));
  }else{
    return(NULL);
  }
}

static PyObject *create_esi_idw(PyObject *self, PyObject *args){
  PyArrayObject *samples, *values, *bbox;
  char *path;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_bbx;
  std::vector<float> c_val;
  int forest_size, seed;
  float alpha, exp;

  // parse arguments
  if (!PyArg_ParseTuple(args, "sO!O!ifO!fi", &path, &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &PyArray_Type, &bbox, &exp, &seed)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return((PyObject *) NULL);
  }

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return((PyObject *) NULL);
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return((PyObject *) NULL);
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  npy_intp *bbx_sh = PyArray_SHAPE(bbox);
  if (bbx_sh[0]!=smp_sh[1]){
    PyErr_SetString(PyExc_TypeError, "[6] bbox should have same elements per row as samples");
    return((PyObject *) NULL);
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_val.push_back(aux[j*smp_sh[0]+i]);
      }
      c_smp.push_back(c_val);
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_val.push_back(aux[smp_sh[1]*i+j]);
      }
      c_smp.push_back(c_val);
    }
  }
  aux = (float *)PyArray_DATA(bbox);
  if (PyArray_CHKFLAGS(bbox, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<bbx_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<bbx_sh[1]; j++){
        c_val.push_back(aux[j*bbx_sh[0]+i]);
      }
      c_bbx.push_back(c_val);
    }
  }else{
    for(int i=0; i<bbx_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<bbx_sh[1]; j++){
        c_val.push_back(aux[bbx_sh[1]*i+j]);
      }
      c_bbx.push_back(c_val);
    }
  }
  aux = (float *)PyArray_DATA(values);
  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));

  // ##### THE METHOD ITSELF #####
  new sptlz::ESI_IDW(std::string(path), &c_smp, &c_val, forest_size, alpha, sptlz::as_1d_array(&c_bbx), exp, seed);

  Py_RETURN_NONE;
}

static PyObject *create_esi_kriging(PyObject *self, PyObject *args){
  PyArrayObject *samples, *values, *bbox;
  char *path;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_bbx;
  std::vector<float> c_val;
  int forest_size, seed, model;
  float alpha, nugget, range;

  // parse arguments
  if (!PyArg_ParseTuple(args, "sO!O!ifO!iffi", &path, &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &PyArray_Type, &bbox, &model, &nugget, &range, &seed)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return((PyObject *) NULL);
  }

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return((PyObject *) NULL);
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return((PyObject *) NULL);
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  npy_intp *bbx_sh = PyArray_SHAPE(bbox);
  if (bbx_sh[0]!=smp_sh[1]){
    PyErr_SetString(PyExc_TypeError, "[6] bbox should have same elements per row as samples");
    return((PyObject *) NULL);
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_val.push_back(aux[j*smp_sh[0]+i]);
      }
      c_smp.push_back(c_val);
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_val.push_back(aux[smp_sh[1]*i+j]);
      }
      c_smp.push_back(c_val);
    }
  }
  aux = (float *)PyArray_DATA(bbox);
  if (PyArray_CHKFLAGS(bbox, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<bbx_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<bbx_sh[1]; j++){
        c_val.push_back(aux[j*bbx_sh[0]+i]);
      }
      c_bbx.push_back(c_val);
    }
  }else{
    for(int i=0; i<bbx_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<bbx_sh[1]; j++){
        c_val.push_back(aux[bbx_sh[1]*i+j]);
      }
      c_bbx.push_back(c_val);
    }
  }
  aux = (float *)PyArray_DATA(values);
  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));

  // ##### THE METHOD ITSELF #####
  new sptlz::ESI_Kriging(std::string(path), &c_smp, &c_val, forest_size, alpha, sptlz::as_1d_array(&c_bbx), model, nugget, range, seed);

  Py_RETURN_NONE;
}

static PyObject *estimation_stored_model(PyObject *self, PyObject *args){
  PyObject *func, *aux_str;
  PyArrayObject *scattered;
  char *path;
  float *aux;
  std::vector<std::vector<float>> c_loc, r;
  std::vector<float> c_val;
  int has_call;
  std::string fname;
  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "sO!O", &path, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return((PyObject *) NULL);
  }

  // check visitor function
  has_call = PyObject_HasAttrString(func, "__call__");
  if(!has_call){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return((PyObject *) NULL);
  }

  // Check if C contiguous data (if not we should transpose)
  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<sct_sh[1]; j++){
        c_val.push_back(aux[j*sct_sh[0]+i]);
      }
      c_loc.push_back(c_val);
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<sct_sh[1]; j++){
        c_val.push_back(aux[sct_sh[1]*i+j]);
      }
      c_loc.push_back(c_val);
    }
  }

  // ##### THE METHOD ITSELF #####
  sptlz::ESI *esi = get_esi_model(path);

  r = esi->estimate(&c_loc, [func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  });
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), (int)r.at(0).size()};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *) estimation);
}

static PyObject *loo_stored_model(PyObject *self, PyObject *args){
  PyObject *func, *aux_str;
  char *path;
  float *aux;
  std::vector<std::vector<float>> r;
  int has_call;
  std::string fname;
  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "sO", &path, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return((PyObject *) NULL);
  }

  // check visitor function
  has_call = PyObject_HasAttrString(func, "__call__");
  if(!has_call){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // ##### THE METHOD ITSELF #####
  sptlz::ESI *esi = get_esi_model(path);

  // make loo validation
  r = esi->leave_one_out([func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  });
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), (int)r.at(0).size()};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *) estimation);
}

static PyObject *kfold_stored_model(PyObject *self, PyObject *args){
  PyObject *func, *aux_str;
  char *path;
  float *aux;
  std::vector<std::vector<float>> r;
  int has_call, k,folding_seed;
  std::string fname;
  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "siiO", &path, &k, &folding_seed, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return((PyObject *) NULL);
  }

  // check visitor function
  has_call = PyObject_HasAttrString(func, "__call__");
  if(!has_call){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // ##### THE METHOD ITSELF #####
  sptlz::ESI *esi = get_esi_model(path);

  // make loo validation
  r = esi->k_fold(k, [func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  }, folding_seed);
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), (int)r.at(0).size()};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *) estimation);
}

static PyObject *estimation_esi_idw(PyObject *self, PyObject *args){
  PyObject *func, *aux_str;
  PyArrayObject *samples, *values, *scattered;
  char *path;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc, r;
  std::vector<float> c_val;
  int forest_size, has_call, seed;
  float alpha, exp;
  std::string fname;
  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "sO!O!iffiO!O", &path, &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &exp, &seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return((PyObject *) NULL);
  }

  // check visitor function
  has_call = PyObject_HasAttrString(func, "__call__");
  if(!has_call){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return((PyObject *) NULL);
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return((PyObject *) NULL);
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return((PyObject *) NULL);
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=smp_sh[1]){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have same elements per row as samples");
    return((PyObject *) NULL);
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_val.push_back(aux[j*smp_sh[0]+i]);
      }
      c_smp.push_back(c_val);
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_val.push_back(aux[smp_sh[1]*i+j]);
      }
      c_smp.push_back(c_val);
    }
  }
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<sct_sh[1]; j++){
        c_val.push_back(aux[j*sct_sh[0]+i]);
      }
      c_loc.push_back(c_val);
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<sct_sh[1]; j++){
        c_val.push_back(aux[sct_sh[1]*i+j]);
      }
      c_loc.push_back(c_val);
    }
  }
  aux = (float *)PyArray_DATA(values);
  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::coords_bbox(&c_loc);
  auto bbox2 = sptlz::coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(2*i) < bbox.at(2*i)){bbox.at(2*i) = bbox2.at(2*i);}
    if(bbox2.at(2*i+1) > bbox.at(2*i+1)){bbox.at(2*i+1) = bbox2.at(2*i+1);}
  }
  sptlz::ESI_IDW* esi = new sptlz::ESI_IDW(std::string(path), &c_smp, &c_val, forest_size, alpha, bbox, exp, seed);

  // make the estimation
  r = esi->estimate(&c_loc, [func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  });
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *) estimation);
}

static PyObject *loo_esi_idw(PyObject *self, PyObject *args){
  PyObject *func, *aux_str;
  PyArrayObject *samples, *values, *scattered;
  char *path;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc, r;
  std::vector<float> c_val;
  int forest_size, has_call, seed;
  float alpha, exp;
  std::string fname;
  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "sO!O!iffiO!O", &path, &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &exp, &seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return((PyObject *) NULL);
  }

  // check visitor function
  has_call = PyObject_HasAttrString(func, "__call__");
  if(!has_call){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return((PyObject *) NULL);
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return((PyObject *) NULL);
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return((PyObject *) NULL);
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=smp_sh[1]){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have same elements per row as samples");
    return((PyObject *) NULL);
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_val.push_back(aux[j*smp_sh[0]+i]);
      }
      c_smp.push_back(c_val);
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_val.push_back(aux[smp_sh[1]*i+j]);
      }
      c_smp.push_back(c_val);
    }
  }
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<sct_sh[1]; j++){
        c_val.push_back(aux[j*sct_sh[0]+i]);
      }
      c_loc.push_back(c_val);
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<sct_sh[1]; j++){
        c_val.push_back(aux[sct_sh[1]*i+j]);
      }
      c_loc.push_back(c_val);
    }
  }
  aux = (float *)PyArray_DATA(values);
  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::coords_bbox(&c_loc);
  auto bbox2 = sptlz::coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(2*i) < bbox.at(2*i)){bbox.at(2*i) = bbox2.at(2*i);}
    if(bbox2.at(2*i+1) > bbox.at(2*i+1)){bbox.at(2*i+1) = bbox2.at(2*i+1);}
  }
  sptlz::ESI_IDW* esi = new sptlz::ESI_IDW(std::string(path), &c_smp, &c_val, forest_size, alpha, bbox, exp, seed);

  // make loo validation
  r = esi->leave_one_out([func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  });
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *) estimation);
}

static PyObject *kfold_esi_idw(PyObject *self, PyObject *args){
  PyObject *func, *aux_str;
  PyArrayObject *samples, *values, *scattered;
  char *path;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc, r;
  std::vector<float> c_val;
  int forest_size, has_call, k, creation_seed, folding_seed;
  float alpha, exp;
  std::string fname;
  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "sO!O!iffiiiO!O", &path, &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &exp, &creation_seed, &k, &folding_seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return((PyObject *) NULL);
  }

  // check visitor function
  has_call = PyObject_HasAttrString(func, "__call__");
  if(!has_call){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return((PyObject *) NULL);
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return((PyObject *) NULL);
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return((PyObject *) NULL);
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=smp_sh[1]){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have same elements per row as samples");
    return((PyObject *) NULL);
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_val.push_back(aux[j*smp_sh[0]+i]);
      }
      c_smp.push_back(c_val);
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_val.push_back(aux[smp_sh[1]*i+j]);
      }
      c_smp.push_back(c_val);
    }
  }
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<sct_sh[1]; j++){
        c_val.push_back(aux[j*sct_sh[0]+i]);
      }
      c_loc.push_back(c_val);
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<sct_sh[1]; j++){
        c_val.push_back(aux[sct_sh[1]*i+j]);
      }
      c_loc.push_back(c_val);
    }
  }
  aux = (float *)PyArray_DATA(values);
  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::coords_bbox(&c_loc);
  auto bbox2 = sptlz::coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(2*i) < bbox.at(2*i)){bbox.at(2*i) = bbox2.at(2*i);}
    if(bbox2.at(2*i+1) > bbox.at(2*i+1)){bbox.at(2*i+1) = bbox2.at(2*i+1);}
  }
  sptlz::ESI_IDW* esi = new sptlz::ESI_IDW(std::string(path), &c_smp, &c_val, forest_size, alpha, bbox, exp, creation_seed);

  // make loo validation
  r = esi->k_fold(k, [func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  }, folding_seed);
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *) estimation);
}

static PyObject *estimation_esi_kriging(PyObject *self, PyObject *args){
  PyObject *func, *aux_str;
  PyArrayObject *samples, *values, *scattered;
  char *path;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc, r;
  std::vector<float> c_val;
  int forest_size, has_call, seed, model;
  float alpha, nugget, range;
  std::string fname;
  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "sO!O!ififfiO!O", &path, &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &model, &nugget, &range, &seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return((PyObject *) NULL);
  }

  // check visitor function
  has_call = PyObject_HasAttrString(func, "__call__");
  if(!has_call){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return((PyObject *) NULL);
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return((PyObject *) NULL);
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return((PyObject *) NULL);
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=smp_sh[1]){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have same elements per row as samples");
    return((PyObject *) NULL);
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_val.push_back(aux[j*smp_sh[0]+i]);
      }
      c_smp.push_back(c_val);
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_val.push_back(aux[smp_sh[1]*i+j]);
      }
      c_smp.push_back(c_val);
    }
  }
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<sct_sh[1]; j++){
        c_val.push_back(aux[j*sct_sh[0]+i]);
      }
      c_loc.push_back(c_val);
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<sct_sh[1]; j++){
        c_val.push_back(aux[sct_sh[1]*i+j]);
      }
      c_loc.push_back(c_val);
    }
  }
  aux = (float *)PyArray_DATA(values);
  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::coords_bbox(&c_loc);
  auto bbox2 = sptlz::coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(2*i) < bbox.at(2*i)){bbox.at(2*i) = bbox2.at(2*i);}
    if(bbox2.at(2*i+1) > bbox.at(2*i+1)){bbox.at(2*i+1) = bbox2.at(2*i+1);}
  }
  sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(std::string(path), &c_smp, &c_val, forest_size, alpha, bbox, model, nugget, range, seed);

  // make the estimation
  r = esi->estimate(&c_loc, [func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  });
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *) estimation);
}

static PyObject *loo_esi_kriging(PyObject *self, PyObject *args){
  PyObject *func, *aux_str;
  PyArrayObject *samples, *values, *scattered;
  char *path;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc, r;
  std::vector<float> c_val;
  int forest_size, has_call, seed, model;
  float alpha, nugget, range;
  std::string fname;
  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "sO!O!ififfiO!O", &path, &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &model, &nugget, &range, &seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return((PyObject *) NULL);
  }

  // check visitor function
  has_call = PyObject_HasAttrString(func, "__call__");
  if(!has_call){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return((PyObject *) NULL);
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return((PyObject *) NULL);
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return((PyObject *) NULL);
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=smp_sh[1]){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have same elements per row as samples");
    return((PyObject *) NULL);
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_val.push_back(aux[j*smp_sh[0]+i]);
      }
      c_smp.push_back(c_val);
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_val.push_back(aux[smp_sh[1]*i+j]);
      }
      c_smp.push_back(c_val);
    }
  }
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<sct_sh[1]; j++){
        c_val.push_back(aux[j*sct_sh[0]+i]);
      }
      c_loc.push_back(c_val);
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<sct_sh[1]; j++){
        c_val.push_back(aux[sct_sh[1]*i+j]);
      }
      c_loc.push_back(c_val);
    }
  }
  aux = (float *)PyArray_DATA(values);
  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::coords_bbox(&c_loc);
  auto bbox2 = sptlz::coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(2*i) < bbox.at(2*i)){bbox.at(2*i) = bbox2.at(2*i);}
    if(bbox2.at(2*i+1) > bbox.at(2*i+1)){bbox.at(2*i+1) = bbox2.at(2*i+1);}
  }
  sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(std::string(path), &c_smp, &c_val, forest_size, alpha, bbox, model, nugget, range, seed);

  // make loo validation
  r = esi->leave_one_out([func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  });
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *) estimation);
}

static PyObject *kfold_esi_kriging(PyObject *self, PyObject *args){
  PyObject *func, *aux_str;
  PyArrayObject *samples, *values, *scattered;
  char *path;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc, r;
  std::vector<float> c_val;
  int forest_size, has_call, k, creation_seed, folding_seed, model;
  float alpha, nugget, range;
  std::string fname;
  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "sO!O!ififfiiiO!O", &path, &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &model, &nugget, &range, &creation_seed, &k, &folding_seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return((PyObject *) NULL);
  }

  // check visitor function
  has_call = PyObject_HasAttrString(func, "__call__");
  if(!has_call){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return((PyObject *) NULL);
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return((PyObject *) NULL);
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return((PyObject *) NULL);
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=smp_sh[1]){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have same elements per row as samples");
    return((PyObject *) NULL);
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_val.push_back(aux[j*smp_sh[0]+i]);
      }
      c_smp.push_back(c_val);
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_val.push_back(aux[smp_sh[1]*i+j]);
      }
      c_smp.push_back(c_val);
    }
  }
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<sct_sh[1]; j++){
        c_val.push_back(aux[j*sct_sh[0]+i]);
      }
      c_loc.push_back(c_val);
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_val.clear();
      for(int j=0; j<sct_sh[1]; j++){
        c_val.push_back(aux[sct_sh[1]*i+j]);
      }
      c_loc.push_back(c_val);
    }
  }
  aux = (float *)PyArray_DATA(values);
  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::coords_bbox(&c_loc);
  auto bbox2 = sptlz::coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(2*i) < bbox.at(2*i)){bbox.at(2*i) = bbox2.at(2*i);}
    if(bbox2.at(2*i+1) > bbox.at(2*i+1)){bbox.at(2*i+1) = bbox2.at(2*i+1);}
  }
  sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(std::string(path), &c_smp, &c_val, forest_size, alpha, bbox, model, nugget, range, creation_seed);

  // make loo validation
  r = esi->k_fold(k, [func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  }, folding_seed);
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *) estimation);
}

// THE MODULE'S STUFF
static PyMethodDef SpatialiteMethods[] = {
  // model creation, no estimation, loo or kfold
  { "create_esi_idw", create_esi_idw, METH_VARARGS, "Create ESI_IDW model" },
  { "create_esi_kriging", create_esi_kriging, METH_VARARGS, "Create ESI_Kriging model" },
  // stored model
  { "estimation_stored_model", estimation_stored_model, METH_VARARGS, "Use stored model to estimate" },
  { "loo_stored_model", loo_stored_model, METH_VARARGS, "Use stored model to leave-one-out validation" },
  { "kfold_stored_model", kfold_stored_model, METH_VARARGS, "Use stored model to k-fold validation" },
  // idw
  { "estimation_esi_idw", estimation_esi_idw, METH_VARARGS, "Use ESI_IDW to estimate" },
  { "loo_esi_idw", loo_esi_idw, METH_VARARGS, "Use ESI_IDW to leave-one-out validation" },
  { "kfold_esi_idw", kfold_esi_idw, METH_VARARGS, "Use ESI_IDW to k-fold validation" },
  // kriging
  { "estimation_esi_kriging", estimation_esi_kriging, METH_VARARGS, "Use ESI_Kriging to estimate" },
  { "loo_esi_kriging", loo_esi_kriging, METH_VARARGS, "Use ESI_Kriging to leave-one-out validation" },
  { "kfold_esi_kriging", kfold_esi_kriging, METH_VARARGS, "Use ESI_Kriging to k-fold validation" },

  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef libspatialite = {
    PyModuleDef_HEAD_INIT,
    "libspatialite",   /* name of module */
    "Python wrapper for C++/SQLite Ensemble Spatial Interpolation library", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SpatialiteMethods
};

PyMODINIT_FUNC PyInit_libspatialite(void){
    PyObject *m = PyModule_Create(&libspatialite);
    if (m == NULL)
        return(NULL);

    // /* Load 'numpy¿ functionality. */
    import_array();
    return(m);
}
