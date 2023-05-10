#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include "spatialize/esi_idw.hpp"
#include "spatialize/esi_idw_anis.hpp"
#include "spatialize/esi_kriging.hpp"

static PyObject *esi_idw_2d(PyObject *self, PyObject *args){
  std::stringstream ss;

  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size;
  float alpha, exp;

  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!iffO!", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &exp, &PyArray_Type, &scattered)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[2] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[3] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[4] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[5] samples should have 2 elements per row (x & y)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have 2 elements per row (x & y)");
    return (PyObject *) NULL;
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[i], aux[smp_sh[0]+i]});
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[2*i], aux[2*i+1]});
    }
  }
  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[i], aux[sct_sh[0]+i]});
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[2*i], aux[2*i+1]});
    }
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_IDW* esi = new sptlz::ESI_IDW(c_smp, c_val, lambda, forest_size, bbox, exp, 2069.36);
  auto r = esi->estimate(&c_loc);
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *)estimation);
}

static PyObject *loo_esi_idw_2d(PyObject *self, PyObject *args){
  std::stringstream ss;

  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size;
  float alpha, exp;

  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!iffO!", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &exp, &PyArray_Type, &scattered)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[2] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[3] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[4] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[5] samples should have 2 elements per row (x & y)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have 2 elements per row (x & y)");
    return (PyObject *) NULL;
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[i], aux[smp_sh[0]+i]});
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[2*i], aux[2*i+1]});
    }
  }
  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[i], aux[sct_sh[0]+i]});
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[2*i], aux[2*i+1]});
    }
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_IDW* esi = new sptlz::ESI_IDW(c_smp, c_val, lambda, forest_size, bbox, exp, 2069.36);
  auto r = esi->leave_one_out();
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *)estimation);
}

static PyObject *kfold_esi_idw_2d(PyObject *self, PyObject *args){
  std::stringstream ss;

  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, k;
  float alpha, exp;

  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!iffiO!", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &exp, &k, &PyArray_Type, &scattered)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[2] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[3] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[4] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[5] samples should have 2 elements per row (x & y)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have 2 elements per row (x & y)");
    return (PyObject *) NULL;
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[i], aux[smp_sh[0]+i]});
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[2*i], aux[2*i+1]});
    }
  }
  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[i], aux[sct_sh[0]+i]});
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[2*i], aux[2*i+1]});
    }
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_IDW* esi = new sptlz::ESI_IDW(c_smp, c_val, lambda, forest_size, bbox, exp, 2069.36);
  auto r = esi->k_fold(k);
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *)estimation);
}

static PyObject *esi_idw_3d(PyObject *self, PyObject *args){
  std::stringstream ss;

  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size;
  float alpha, exp;

  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!iffO!", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &exp, &PyArray_Type, &scattered)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[2] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[3] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[4] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[5] samples should have 3 elements per row (x, y & z)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have 2 elements per row (x, y & z)");
    return (PyObject *) NULL;
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[i], aux[smp_sh[0]+i], aux[2*smp_sh[0]+i]});
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[3*i], aux[3*i+1], aux[3*i+2]});
    }
  }
  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[i], aux[sct_sh[0]+i], aux[2*sct_sh[0]+i]});
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[3*i], aux[3*i+1], aux[3*i+2]});
    }
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_IDW* esi = new sptlz::ESI_IDW(c_smp, c_val, lambda, forest_size, bbox, exp, 2069.36);
  auto r = esi->estimate(&c_loc);
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *)estimation);
}

static PyObject *loo_esi_idw_3d(PyObject *self, PyObject *args){
  std::stringstream ss;

  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size;
  float alpha, exp;

  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!iffO!", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &exp, &PyArray_Type, &scattered)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[2] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[3] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[4] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[5] samples should have 3 elements per row (x, y & z)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have 2 elements per row (x, y & z)");
    return (PyObject *) NULL;
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[i], aux[smp_sh[0]+i], aux[2*smp_sh[0]+i]});
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[3*i], aux[3*i+1], aux[3*i+2]});
    }
  }
  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[i], aux[sct_sh[0]+i], aux[2*sct_sh[0]+i]});
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[3*i], aux[3*i+1], aux[3*i+2]});
    }
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_IDW* esi = new sptlz::ESI_IDW(c_smp, c_val, lambda, forest_size, bbox, exp, 2069.36);
  auto r = esi->leave_one_out();
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *)estimation);
}

static PyObject *kfold_esi_idw_3d(PyObject *self, PyObject *args){
  std::stringstream ss;

  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, k;
  float alpha, exp;

  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!iffiO!", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &exp, &k, &PyArray_Type, &scattered)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[2] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[3] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[4] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[5] samples should have 3 elements per row (x, y & z)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have 2 elements per row (x, y & z)");
    return (PyObject *) NULL;
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[i], aux[smp_sh[0]+i], aux[2*smp_sh[0]+i]});
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[3*i], aux[3*i+1], aux[3*i+2]});
    }
  }
  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[i], aux[sct_sh[0]+i], aux[2*sct_sh[0]+i]});
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[3*i], aux[3*i+1], aux[3*i+2]});
    }
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_IDW* esi = new sptlz::ESI_IDW(c_smp, c_val, lambda, forest_size, bbox, exp, 2069.36);
  auto r = esi->k_fold(k);
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *)estimation);
}

static PyObject *esi_kriging_2d(PyObject *self, PyObject *args){
  std::stringstream ss;

  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, model;
  float alpha, nugget, range;

  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ififfO!", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, & model, &nugget, &range, &PyArray_Type, &scattered)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[2] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[3] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[4] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[5] samples should have 2 elements per row (x & y)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have 2 elements per row (x & y)");
    return (PyObject *) NULL;
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[i], aux[smp_sh[0]+i]});
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[2*i], aux[2*i+1]});
    }
  }
  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[i], aux[sct_sh[0]+i]});
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[2*i], aux[2*i+1]});
    }
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(c_smp, c_val, lambda, forest_size, bbox, model, nugget, range, 2069.36);
  auto r = esi->estimate(&c_loc);
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *)estimation);
}

static PyObject *loo_esi_kriging_2d(PyObject *self, PyObject *args){
  std::stringstream ss;

  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, model;
  float alpha, nugget, range;

  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ififfO!", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, & model, &nugget, &range, &PyArray_Type, &scattered)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[2] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[3] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[4] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[5] samples should have 2 elements per row (x & y)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have 2 elements per row (x & y)");
    return (PyObject *) NULL;
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[i], aux[smp_sh[0]+i]});
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[2*i], aux[2*i+1]});
    }
  }
  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[i], aux[sct_sh[0]+i]});
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[2*i], aux[2*i+1]});
    }
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(c_smp, c_val, lambda, forest_size, bbox, model, nugget, range, 2069.36);
  auto r = esi->leave_one_out();
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *)estimation);
}

static PyObject *kfold_esi_kriging_2d(PyObject *self, PyObject *args){
  std::stringstream ss;

  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, model, k;
  float alpha, nugget, range;

  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ififfiO!", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, & model, &nugget, &range, &k, &PyArray_Type, &scattered)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[2] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[3] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[4] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[5] samples should have 2 elements per row (x & y)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have 2 elements per row (x & y)");
    return (PyObject *) NULL;
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[i], aux[smp_sh[0]+i]});
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[2*i], aux[2*i+1]});
    }
  }
  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[i], aux[sct_sh[0]+i]});
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[2*i], aux[2*i+1]});
    }
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(c_smp, c_val, lambda, forest_size, bbox, model, nugget, range, 2069.36);
  auto r = esi->k_fold(k);
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *)estimation);
}

static PyObject *esi_kriging_3d(PyObject *self, PyObject *args){
  std::stringstream ss;

  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, model;
  float alpha, nugget, range;

  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ififfO!", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, & model, &nugget, &range, &PyArray_Type, &scattered)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[2] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[3] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[4] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[5] samples should have 3 elements per row (x, y & z)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have 2 elements per row (x, y & z)");
    return (PyObject *) NULL;
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[i], aux[smp_sh[0]+i], aux[2*smp_sh[0]+i]});
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[3*i], aux[3*i+1], aux[3*i+2]});
    }
  }
  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[i], aux[sct_sh[0]+i], aux[2*sct_sh[0]+i]});
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[3*i], aux[3*i+1], aux[3*i+2]});
    }
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(c_smp, c_val, lambda, forest_size, bbox, model, nugget, range, 2069.36);
  auto r = esi->estimate(&c_loc);
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *)estimation);
}

static PyObject *loo_esi_kriging_3d(PyObject *self, PyObject *args){
  std::stringstream ss;

  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, model;
  float alpha, nugget, range;

  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ififfO!", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, & model, &nugget, &range, &PyArray_Type, &scattered)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[2] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[3] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[4] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[5] samples should have 3 elements per row (x, y & z)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have 2 elements per row (x, y & z)");
    return (PyObject *) NULL;
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[i], aux[smp_sh[0]+i], aux[2*smp_sh[0]+i]});
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[3*i], aux[3*i+1], aux[3*i+2]});
    }
  }
  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[i], aux[sct_sh[0]+i], aux[2*sct_sh[0]+i]});
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[3*i], aux[3*i+1], aux[3*i+2]});
    }
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(c_smp, c_val, lambda, forest_size, bbox, model, nugget, range, 2069.36);
  auto r = esi->leave_one_out();
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *)estimation);
}

static PyObject *kfold_esi_kriging_3d(PyObject *self, PyObject *args){
  std::stringstream ss;

  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, model, k;
  float alpha, nugget, range;

  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ififfiO!", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, & model, &nugget, &range, &k, &PyArray_Type, &scattered)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[2] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[3] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[4] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[5] samples should have 3 elements per row (x, y & z)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have 2 elements per row (x, y & z)");
    return (PyObject *) NULL;
  }

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[i], aux[smp_sh[0]+i], aux[2*smp_sh[0]+i]});
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_smp.push_back({aux[3*i], aux[3*i+1], aux[3*i+2]});
    }
  }
  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  aux = (float *)PyArray_DATA(scattered);
  if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[i], aux[sct_sh[0]+i], aux[2*sct_sh[0]+i]});
    }
  }else{
    for(int i=0; i<sct_sh[0]; i++){
      c_loc.push_back({aux[3*i], aux[3*i+1], aux[3*i+2]});
    }
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(c_smp, c_val, lambda, forest_size, bbox, model, nugget, range, 2069.36);
  auto r = esi->k_fold(k);
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  return((PyObject *)estimation);
}

static PyMethodDef SpatializeMethods[] = {
  { "esi_idw_2d", esi_idw_2d, METH_VARARGS, "Esi using IDW on 2 dimensions" },
  { "loo_esi_idw_2d", loo_esi_idw_2d, METH_VARARGS, "Leave-one-out validation for Esi using IDW on 2 dimensions" },
  { "kfold_esi_idw_2d", kfold_esi_idw_2d, METH_VARARGS, "K-fold validation for Esi using IDW on 2 dimensions" },
  { "esi_idw_3d", esi_idw_3d, METH_VARARGS, "Esi using IDW on 3 dimensions" },
  { "loo_esi_idw_3d", loo_esi_idw_3d, METH_VARARGS, "Leave-one-out validation for Esi using IDW on 3 dimensions" },
  { "kfold_esi_idw_3d", kfold_esi_idw_3d, METH_VARARGS, "K-fold validation for Esi using IDW on 3 dimensions" },
  { "esi_kriging_2d", esi_kriging_2d, METH_VARARGS, "Esi using Kriging on 2 dimensions" },
  { "loo_esi_kriging_2d", loo_esi_kriging_2d, METH_VARARGS, "Leave-one-out validation for Esi using Kriging on 2 dimensions" },
  { "kfold_esi_kriging_2d", kfold_esi_kriging_2d, METH_VARARGS, "K-fold validation for Esi using Kriging on 2 dimensions" },
  { "esi_kriging_3d", esi_kriging_3d, METH_VARARGS, "Esi using Kriging on 3 dimensions" },
  { "loo_esi_kriging_3d", loo_esi_kriging_3d, METH_VARARGS, "Leave-one-out validation for Esi using Kriging on 3 dimensions" },
  { "kfold_esi_kriging_3d", kfold_esi_kriging_3d, METH_VARARGS, "K-fold validation for Esi using Kriging on 3 dimensions" },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef spatialize = {
    PyModuleDef_HEAD_INIT,
    "spatialize",   /* name of module */
    "Python wrapper for c++ ESI library", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SpatializeMethods
};

PyMODINIT_FUNC PyInit_spatialize(void){
    PyObject *m = PyModule_Create(&spatialize);
    if (m == NULL)
        return(NULL);

    // /* Load `numpy` functionality. */
    import_array();
    return(m);
}
