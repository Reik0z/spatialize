#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <iostream>
#include "spatialize/nn_idw.hpp"
#include "spatialize/esi_idw.hpp"
#include "spatialize/esi_kriging.hpp"
#include "spatialize/voronoi_idw.hpp"

// c to py
// static PyObject *float2d_to_list(std::vector<std::vector<float>> *f2d){
//   PyObject *list = Py_BuildValue("[]"), *aux;
//
//   for(auto f1d: *f2d){
//     aux = Py_BuildValue("[]");
//     for(auto v: f1d){
//       PyList_Append(aux, Py_BuildValue("f", v));
//     }
//     PyList_Append(list, aux);
//   }
//   return(list);
// }

// static PyObject *int2d_to_list(std::vector<std::vector<int>> *i2d){
//   PyObject *list = Py_BuildValue("[]"), *aux;
//
//   for(auto i1d: *i2d){
//     aux = Py_BuildValue("[]");
//     for(auto v: i1d){
//       PyList_Append(aux, Py_BuildValue("i", v));
//     }
//     PyList_Append(list, aux);
//   }
//   return(list);
// }

// static PyObject *float1d_to_list(std::vector<float> *f1d){
//   PyObject *list = Py_BuildValue("[]");
//   for(auto v: *f1d){
//     PyList_Append(list, Py_BuildValue("f", v));
//   }
//   return(list);
// }

// static PyObject *int1d_to_list(std::vector<int> *i1d){
//   PyObject *list = Py_BuildValue("[]");
//   for(auto v: *i1d){
//     PyList_Append(list, Py_BuildValue("i", v));
//   }
//   return(list);
// }

// static PyObject *nodetree_as_dict(sptlz::MondrianNode* node){
//   if(node==NULL){
//     Py_RETURN_NONE;
//   }else{
//     PyObject *dict = Py_BuildValue("{s:O, s:O, s:O, s:O, s:O, s:O, s:O, s:O}",
//       "leaf_id", Py_BuildValue("i", node->leaf_id),
//       "bbox", float2d_to_list(&(node->bbox)),
//       "tau", Py_BuildValue("f", node->tau),
//       "cut", Py_BuildValue("f", node->cut),
//       "height", Py_BuildValue("i", node->height),
//       "axis", Py_BuildValue("i", node->axis),
//       "left", nodetree_as_dict(node->left),
//       "right", nodetree_as_dict(node->right)
//     );
//     return(dict);
//   }
// }

// static PyObject *nodelist_to_idlist(std::vector<sptlz::MondrianNode*> *nodes){
//   PyObject *list = Py_BuildValue("[]"), *aux;
//   std::ignore = aux;
//
//   for(auto node: *nodes){
//     PyList_Append(list, Py_BuildValue("i", node->leaf_id));
//   }
//   return(list);
// }

// static PyObject *tree_as_dict(sptlz::ESI *esi, int i){
//   sptlz::MondrianTree *t = esi->get_tree(i);
//   PyObject *dict = Py_BuildValue("{s:O, s:O, s:O, s:O, s:O, s:O}",
//     "root", nodetree_as_dict(t->root),
//     "leaves_id", nodelist_to_idlist(&(t->leaves)),
//     "leaf_for_sample", int1d_to_list(&(t->leaf_for_sample)),
//     "samples_by_leaf", int2d_to_list(&(t->samples_by_leaf)),
//     "leaf_params", float2d_to_list(&(t->leaf_params)),
//     "ndim", Py_BuildValue("i", t->ndim)
//   );
//
//   return(dict);
// }

// static PyObject *esi_idw_to_dict(sptlz::ESI_IDW *esi){
//   PyObject *list = Py_BuildValue("[]");
//   for(int i=0; i<esi->forest_size(); i++){
//     PyList_Append(list, tree_as_dict(esi, i));
//   }
//
//   PyObject *dict = Py_BuildValue("{s:i, s:O, s:O, s:O, s:O}",
//     "esi_type", 1,
//     "coords", float2d_to_list(esi->get_coords()),
//     "values", float1d_to_list(esi->get_values()),
//     "mondrian_forest", list,
//     "exponent", Py_BuildValue("f", esi->get_exponent())
//   );
//
//   return(dict);
// }
//
//
// static PyObject *esi_kriging_to_dict(sptlz::ESI_Kriging *esi){
//   PyObject *list = Py_BuildValue("[]");
//   for(int i=0; i<esi->forest_size(); i++){
//     PyList_Append(list, tree_as_dict(esi, i));
//   }
//
//   PyObject *dict = Py_BuildValue("{s:i, s:O, s:O, s:O, s:O, s:O, s:O}",
//     "esi_type", 2,
//     "coords", float2d_to_list(esi->get_coords()),
//     "values", float1d_to_list(esi->get_values()),
//     "mondrian_forest", list,
//     "variogram_model", Py_BuildValue("i", esi->get_variogram_model()),
//     "range", Py_BuildValue("f", esi->get_range()),
//     "nugget", Py_BuildValue("f", esi->get_nugget())
//   );
//   return(dict);
// }

// py to c
std::vector<std::vector<float>> list_to_float2d(PyObject* po){
  PyObject* aux;
  std::vector<float> coords;

  int n = PyList_Size(po), m;
  std::vector<std::vector<float>> result = std::vector<std::vector<float>>(n);
  for(int i=0; i<n; i++){
    aux = PyList_GetItem(po, i);
    m = PyList_Size(aux);
    coords = std::vector<float>(m);
    for(int j=0; j<m; j++){
      coords.at(j) = (float)PyFloat_AsDouble(PyList_GetItem(aux, j));
    }
    result.at(i) = coords;
  }

  return(result);
}

std::vector<float> list_to_float1d(PyObject* po){
  int n = PyList_Size(po);
  std::vector<float> result = std::vector<float>(n);
  for(int i=0; i<n; i++){
    result.at(i) = (float)PyFloat_AsDouble(PyList_GetItem(po, i));
  }

  return(result);
}

std::vector<std::vector<int>> list_to_int2d(PyObject* po){
  PyObject* aux;
  std::vector<int> coords;

  int n = PyList_Size(po), m;
  std::vector<std::vector<int>> result = std::vector<std::vector<int>>(n);
  for(int i=0; i<n; i++){
    aux = PyList_GetItem(po, i);
    m = PyList_Size(aux);
    coords = std::vector<int>(m);
    for(int j=0; j<m; j++){
      coords.at(j) = (int)PyLong_AsLong(PyList_GetItem(aux, j));
    }
    result.at(i) = coords;
  }

  return(result);
}

std::vector<int> list_to_int1d(PyObject* po){
  int n = PyList_Size(po);
  std::vector<int> result = std::vector<int>(n);
  for(int i=0; i<n; i++){
    result.at(i) = (int)PyLong_AsLong(PyList_GetItem(po, i));
  }

  return(result);
}

sptlz::MondrianNode *dict_to_tree(PyObject *po){
    if(po==Py_None){
      return(NULL);
    }else{
      sptlz::MondrianNode* mn = new sptlz::MondrianNode();

      mn->leaf_id = (int)PyLong_AsLong(PyDict_GetItem(po, Py_BuildValue("s", "leaf_id")));
      mn->bbox = list_to_float2d(PyDict_GetItem(po, Py_BuildValue("s", "bbox")));
      mn->tau = (float)PyFloat_AsDouble(PyDict_GetItem(po, Py_BuildValue("s", "tau")));
      mn->cut = (float)PyFloat_AsDouble(PyDict_GetItem(po, Py_BuildValue("s", "cut")));
      mn->height = (int)PyLong_AsLong(PyDict_GetItem(po, Py_BuildValue("s", "height")));
      mn->axis = (int)PyLong_AsLong(PyDict_GetItem(po, Py_BuildValue("s", "axis")));
      mn->left = dict_to_tree(PyDict_GetItem(po, Py_BuildValue("s", "left")));
      mn->right = dict_to_tree(PyDict_GetItem(po, Py_BuildValue("s", "right")));
      return(mn);
    }
}

void fill_nodes(sptlz::MondrianNode* node, std::map<int,sptlz::MondrianNode*> *nodemap){
  if(node!=NULL){
    (*nodemap)[node->leaf_id] = node;
    fill_nodes(node->left, nodemap);
    fill_nodes(node->right, nodemap);
  }
}

std::vector<sptlz::MondrianTree*> list_to_mondrian1d(PyObject* po){
  int n = PyList_Size(po);
  std::vector<sptlz::MondrianTree*> mt = std::vector<sptlz::MondrianTree*>(n);
  std::vector<int> ids;
  std::map<int,sptlz::MondrianNode*>::iterator it;
  std::map<int,sptlz::MondrianNode*> nodemap;
  std::vector<sptlz::MondrianNode*> nodes;
  sptlz::MondrianTree* t;
  PyObject *aux;


  for(int i=0; i<n; i++){
    aux = PyList_GetItem(po, i);
    t = new sptlz::MondrianTree();

    t->root = dict_to_tree(PyDict_GetItem(aux, Py_BuildValue("s", "root")));
    ids = list_to_int1d(PyDict_GetItem(aux, Py_BuildValue("s", "leaves_id")));
    nodemap.clear();
    fill_nodes(t->root, &nodemap);
    nodes.clear();
    for(size_t j=0; j<ids.size(); j++){
      it = nodemap.find(ids[j]);
      if(it!=nodemap.end()){
        nodes.push_back(it->second);
      }
    }
    t->leaves = nodes;
    t->leaf_for_sample = list_to_int1d(PyDict_GetItem(aux, Py_BuildValue("s", "leaf_for_sample")));
    t->samples_by_leaf = list_to_int2d(PyDict_GetItem(aux, Py_BuildValue("s", "samples_by_leaf")));
    t->leaf_params = list_to_float2d(PyDict_GetItem(aux, Py_BuildValue("s", "leaf_params")));
    t->ndim = (int)PyLong_AsLong(PyDict_GetItem(aux, Py_BuildValue("s", "ndim")));

    mt.at(i) = t;
  }

  return(mt);
}

static sptlz::ESI *load_esi(PyObject *dict){
  auto esitype = PyLong_AsLong(PyDict_GetItem(dict, Py_BuildValue("s", "esi_type")));
  std::vector<std::vector<float>> coords = list_to_float2d(PyDict_GetItem(dict, Py_BuildValue("s", "coords")));
  std::vector<float> values = list_to_float1d(PyDict_GetItem(dict, Py_BuildValue("s", "values")));
  std::vector<sptlz::MondrianTree*> mondrian_forest = list_to_mondrian1d(PyDict_GetItem(dict, Py_BuildValue("s", "mondrian_forest")));
  
  if(esitype == 1){
    float exponent = PyFloat_AsDouble(PyDict_GetItem(dict, Py_BuildValue("s", "exponent")));
    sptlz::ESI_IDW *esi = new sptlz::ESI_IDW(mondrian_forest, coords, values, exponent);
    return(esi);
  }else if(esitype == 2){
    int var_model = PyLong_AsLong(PyDict_GetItem(dict, Py_BuildValue("s", "variogram_model")));
    float range = PyFloat_AsDouble(PyDict_GetItem(dict, Py_BuildValue("s", "range")));
    float nugget = PyFloat_AsDouble(PyDict_GetItem(dict, Py_BuildValue("s", "nugget")));
  float sill = PyFloat_AsDouble(PyDict_GetItem(dict, Py_BuildValue("s", "sill")));
    sptlz::ESI_Kriging *esi = new sptlz::ESI_Kriging(mondrian_forest, coords, values, var_model, nugget, range, sill);
    return(esi);
  }else{
    return(NULL);
  }
}

// exposed functions
static PyObject *get_partitions_using_esi(PyObject *self, PyObject *args){
  PyArrayObject *samples, *estimation;
  float *aux;
  std::vector<std::vector<float>> c_smp;
  std::vector<std::vector<int>> r;
  std::vector<float> c_aux;
  int forest_size, seed;
  float alpha;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!iff", &PyArray_Type, &samples, &forest_size, &alpha, &seed)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return((PyObject *) NULL);
  }

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[2] samples must be a 2 dimensions array");
    return((PyObject *) NULL);
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);

  // Check if C contiguous data (if not we should transpose)
  aux = (float *)PyArray_DATA(samples);
  if (PyArray_CHKFLAGS(samples, NPY_ARRAY_F_CONTIGUOUS)==1){
    for(int i=0; i<smp_sh[0]; i++){
      c_aux.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_aux.push_back(aux[j*smp_sh[0]+i]);
      }
      c_smp.push_back(c_aux);
    }
  }else{
    for(int i=0; i<smp_sh[0]; i++){
      c_aux.clear();
      for(int j=0; j<smp_sh[1]; j++){
        c_aux.push_back(aux[smp_sh[1]*i+j]);
      }
      c_smp.push_back(c_aux);
    }
  }
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_smp);
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI* esi = new sptlz::ESI(c_smp, {}, lambda, forest_size, bbox, seed);
  r = esi->get_partitions();
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_INT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  delete esi;
  if (Py_REFCNT(esi) != 0) {
      Py_SET_REFCNT(esi, 0);
  }

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  return((PyObject *)estimation);
}

// static PyObject *estimation_using_model(PyObject *self, PyObject *args){
//   PyObject *model, *func, *aux_str;
//   PyArrayObject *scattered, *estimation;
//   int has_call;
//   std::string fname;
//   float *aux;
//   std::vector<std::vector<float>> c_loc, r;
//   std::vector<float> c_aux;

//   // parse arguments
//   if (!PyArg_ParseTuple(args, "OO!O", &model, &PyArray_Type, &scattered, &func)) {
//     PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
//     return((PyObject *) NULL);
//   }

//   has_call = PyObject_HasAttrString(func, "__call__");
//   if(has_call==0){
//     PyErr_SetString(PyExc_TypeError, "[2] Argument do not match");
//     return((PyObject *) NULL);
//   }
//   aux_str = PyObject_GetAttrString(func, "__class__");
//   aux_str = PyObject_GetAttrString(aux_str, "__name__");
//   fname = PyUnicode_AsUTF8(aux_str);

//   // get model
//   sptlz::ESI *esi = load_esi(model);
//   sptlz::MondrianTree *mt = esi->get_tree(0);

//   // Argument validations
//   if (PyArray_NDIM(scattered)!=2){
//     PyErr_SetString(PyExc_TypeError, "[3] scattered must be a 2 dimensions array");
//     return((PyObject *) NULL);
//   }

//   npy_intp *sct_sh = PyArray_SHAPE(scattered);
//   if (sct_sh[1]!=mt->ndim){
//     PyErr_SetString(PyExc_TypeError, "[4] scattered should have same elements per row as samples");
//     return((PyObject *) NULL);
//   }

//   aux = (float *)PyArray_DATA(scattered);
//   if (PyArray_CHKFLAGS(scattered, NPY_ARRAY_F_CONTIGUOUS)==1){
//     for(int i=0; i<sct_sh[0]; i++){
//       c_aux.clear();
//       for(int j=0; j<sct_sh[1]; j++){
//         c_aux.push_back(aux[j*sct_sh[0]+i]);
//       }
//       c_loc.push_back(c_aux);
//     }
//   }else{
//     for(int i=0; i<sct_sh[0]; i++){
//       c_aux.clear();
//       for(int j=0; j<sct_sh[1]; j++){
//         c_aux.push_back(aux[sct_sh[1]*i+j]);
//       }
//       c_loc.push_back(c_aux);
//     }
//   }
//   if (Py_REFCNT(aux) != 0) {
//       Py_SET_REFCNT(aux, 0);
//   }

//   // ##### THE METHOD ITSELF #####
//   r = esi->estimate(&c_loc, [func](std::string s){
//     PyObject *tup = Py_BuildValue("(s)", s.c_str());
//     PyObject_Call(func, tup, NULL);
//     return(0);
//   });
//   auto output = sptlz::as_1d_array(&r);

//   // stuff to return data to python
//   const npy_intp dims[2] = {(int)r.size(), esi->forest_size()};
//   estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
//   aux = (float *)PyArray_DATA(estimation);
//   memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

//   if (Py_REFCNT(aux) != 0) {
//       Py_SET_REFCNT(aux, 0);
//   }

//   return((PyObject *)estimation);
// }

// static PyObject *loo_using_model(PyObject *self, PyObject *args){
//   PyObject *model, *func, *aux_str;
//   PyArrayObject *estimation;
//   int has_call;
//   std::string fname;
//   float *aux;
//   std::vector<std::vector<float>> r;

//   // parse arguments
//   if (!PyArg_ParseTuple(args, "OO", &model, &func)) {
//     PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
//     return((PyObject *) NULL);
//   }

//   has_call = PyObject_HasAttrString(func, "__call__");
//   if(has_call==0){
//     PyErr_SetString(PyExc_TypeError, "[2] Argument do not match");
//     return((PyObject *) NULL);
//   }
//   aux_str = PyObject_GetAttrString(func, "__class__");
//   aux_str = PyObject_GetAttrString(aux_str, "__name__");
//   fname = PyUnicode_AsUTF8(aux_str);

//   // get model
//   sptlz::ESI *esi = load_esi(model);

//   // ##### THE METHOD ITSELF #####
//   r = esi->leave_one_out([func](std::string s){
//     PyObject *tup = Py_BuildValue("(s)", s.c_str());
//     PyObject_Call(func, tup, NULL);
//     return(0);
//   });
//   auto output = sptlz::as_1d_array(&r);

//   // stuff to return data to python
//   const npy_intp dims[2] = {(int)r.size(), esi->forest_size()};
//   estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
//   aux = (float *)PyArray_DATA(estimation);
//   memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

//   if (Py_REFCNT(aux) != 0) {
//       Py_SET_REFCNT(aux, 0);
//   }
//   return((PyObject *)estimation);
// }

// static PyObject *kfold_using_model(PyObject *self, PyObject *args){
//   PyObject *model, *func, *aux_str;
//   PyArrayObject *estimation;
//   int has_call, k, seed;
//   std::string fname;
//   float *aux;
//   std::vector<std::vector<float>> r;

//   // parse arguments
//   if (!PyArg_ParseTuple(args, "OiiO", &model, &k, &seed, &func)) {
//     PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
//     return((PyObject *) NULL);
//   }

//   has_call = PyObject_HasAttrString(func, "__call__");
//   if(has_call==0){
//     PyErr_SetString(PyExc_TypeError, "[2] Argument do not match");
//     return((PyObject *) NULL);
//   }
//   aux_str = PyObject_GetAttrString(func, "__class__");
//   aux_str = PyObject_GetAttrString(aux_str, "__name__");
//   fname = PyUnicode_AsUTF8(aux_str);

//   // get model
//   sptlz::ESI *esi = load_esi(model);

//   // ##### THE METHOD ITSELF #####
//   r = esi->k_fold(k, [func](std::string s){
//     PyObject *tup = Py_BuildValue("(s)", s.c_str());
//     PyObject_Call(func, tup, NULL);
//     return(0);
//   }, seed);
//   auto output = sptlz::as_1d_array(&r);

//   // stuff to return data to python
//   const npy_intp dims[2] = {(int)r.size(), esi->forest_size()};
//   estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
//   aux = (float *)PyArray_DATA(estimation);
//   memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

//   if (Py_REFCNT(aux) != 0) {
//       Py_SET_REFCNT(aux, 0);
//   }
//   return((PyObject *)estimation);
// }

static PyObject *estimation_nn_idw(PyObject *self, PyObject *args){
  PyObject *func, *aux_str;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  float radius, exp;
  int has_call;
  std::string fname;
  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ffO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &radius, &exp, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==0){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=smp_sh[1]){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have same elements per row as samples");
    return (PyObject *) NULL;
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  aux = (float *)PyArray_DATA(values);
  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  // ##### THE METHOD ITSELF #####
  std::vector<float> search_params = {radius, radius, radius, 0.0, 0.0, 0.0};
  sptlz::NN_IDW* myIDW = new sptlz::NN_IDW(c_smp, c_val, search_params, exp);
  auto r = myIDW->estimate(&c_loc, [func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  });

  // stuff to return data to python
  const npy_intp dims[1] = {(int)r.size()};
  estimation = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &r.data()[0], r.size()*sizeof(float));

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  return((PyObject *)estimation);
}

static PyObject *loo_nn_idw(PyObject *self, PyObject *args){
  PyObject *func, *aux_str;
  PyArrayObject *samples, *values;
  float *aux;
  std::vector<std::vector<float>> c_smp;
  std::vector<float> c_val;
  float radius, exp;
  int has_call;
  std::string fname;
  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ffO", &PyArray_Type, &samples, &PyArray_Type, &values, &radius, &exp, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==0){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  aux = (float *)PyArray_DATA(values);
  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  // ##### THE METHOD ITSELF #####
  std::vector<float> search_params = {radius, radius, radius, 0.0, 0.0, 0.0};
  sptlz::NN_IDW* myIDW = new sptlz::NN_IDW(c_smp, c_val, search_params, exp);
  auto r = myIDW->leave_one_out([func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  });

  // stuff to return data to python
  const npy_intp dims[1] = {(int)r.size()};
  estimation = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &r.data()[0], r.size()*sizeof(float));

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }
  return((PyObject *)estimation);
}

static PyObject *kfold_nn_idw(PyObject *self, PyObject *args){
  PyObject *func, *aux_str;
  PyArrayObject *samples, *values;
  float *aux;
  std::vector<std::vector<float>> c_smp;
  std::vector<float> c_val;
  float radius, exp;
  int k, has_call, seed;
  std::string fname;
  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ffiiO", &PyArray_Type, &samples, &PyArray_Type, &values, &radius, &exp, &k, &seed, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==0){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  aux = (float *)PyArray_DATA(values);
  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  // ##### THE METHOD ITSELF #####
  std::vector<float> search_params = {radius, radius, radius, 0.0, 0.0, 0.0};
  sptlz::NN_IDW* myIDW = new sptlz::NN_IDW(c_smp, c_val, search_params, exp);
  auto r = myIDW->k_fold(k, [func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  }, seed);

  // stuff to return data to python
  const npy_intp dims[1] = {(int)r.size()};
  estimation = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &r.data()[0], r.size()*sizeof(float));

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }
  return((PyObject *)estimation);
}

static PyObject *estimation_esi_idw(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc, r;
  std::vector<float> c_val;
  int forest_size, has_call, seed;
  float alpha, exp;
  std::string fname;
  PyArrayObject *estimation;

#ifdef DEBUG
  std::cout << "[C++] parsing arguments" << "\n";
#endif
  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!iffiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &exp, &seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return((PyObject *) NULL);
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==0){
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

#ifdef DEBUG
  std::cout << "[C++] checking data format" << "\n";
#endif
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

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
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

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }
  aux = (float *)PyArray_DATA(values);

  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  // ##### THE METHOD ITSELF #####
#ifdef DEBUG
  std::cout << "[C++] arranging parameters" << "\n";
#endif
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

#ifdef DEBUG
  std::cout << "[C++] building esi" << "\n";
#endif
  sptlz::ESI_IDW* esi = new sptlz::ESI_IDW(c_smp, c_val, lambda, forest_size, bbox, exp, seed);

#ifdef DEBUG
  std::cout << "[C++] calling esi" << "\n";
#endif
  r = esi->estimate(&c_loc, [func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  });

#ifdef DEBUG
  std::cout << "[C++] formatting the output" << "\n";
#endif
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

#ifdef DEBUG
  std::cout << "[C++] building the output model" << "\n";
#endif

  // avoid model construction until we have
  // a more efficient way to pass it through
  //
  // model_list = esi_idw_to_dict(esi);
  model_list = Py_BuildValue("");

  delete esi;

  std::vector<float>().swap(output);
  std::vector<std::vector<float>>().swap(c_smp);
  std::vector<std::vector<float>>().swap(c_loc);
  std::vector<std::vector<float>>().swap(r);
  std::vector<float>().swap(c_val);

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}

static PyObject *loo_esi_idw(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, has_call, seed;
  float alpha, exp;
  std::string fname;
  PyArrayObject *estimation;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!iffiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &exp, &seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==0){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=smp_sh[1]){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have same elements per row as samples");
    return (PyObject *) NULL;
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  aux = (float *)PyArray_DATA(values);
  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_IDW* esi = new sptlz::ESI_IDW(c_smp, c_val, lambda, forest_size, bbox, exp, seed);
  auto r = esi->leave_one_out([func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  });
  auto output = sptlz::as_1d_array(&r);

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  // avoid model construction until we have
  // a more efficient way to pass it through
  //
  // model_list = esi_idw_to_dict(esi);
  model_list = Py_BuildValue("");

  delete esi;

  std::vector<float>().swap(output);
  std::vector<std::vector<float>>().swap(c_smp);
  std::vector<std::vector<float>>().swap(c_loc);
  std::vector<std::vector<float>>().swap(r);
  std::vector<float>().swap(c_val);

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}

static PyObject *kfold_esi_idw(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, has_call, k, creation_seed, folding_seed;
  float alpha, exp;
  std::string fname;
  PyArrayObject *estimation;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!iffiiiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &exp, &creation_seed, &k, &folding_seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==0){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=smp_sh[1]){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have same elements per row as samples");
    return (PyObject *) NULL;
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  aux = (float *)PyArray_DATA(values);
  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_IDW* esi = new sptlz::ESI_IDW(c_smp, c_val, lambda, forest_size, bbox, exp, creation_seed);
  auto r = esi->k_fold(k, [func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  }, folding_seed);
  auto output = sptlz::as_1d_array(&r);

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  // avoid model construction until we have
  // a more efficient way to pass it through
  //
  // model_list = esi_idw_to_dict(esi);
  model_list = Py_BuildValue("");

  delete esi;

  std::vector<float>().swap(output);
  std::vector<std::vector<float>>().swap(c_smp);
  std::vector<std::vector<float>>().swap(c_loc);
  std::vector<std::vector<float>>().swap(r);
  std::vector<float>().swap(c_val);

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}

static PyObject *estimation_esi_kriging_2d(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, has_call, model, seed;
  float alpha, nugget, range, sill;
  std::string fname;
  PyArrayObject *estimation;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ififffiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, & model, &nugget, &range, &sill, &seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==0){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[6] samples should have 2 elements per row (x & y)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[7] scattered should have 2 elements per row (x & y)");
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

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(c_smp, c_val, lambda, forest_size, bbox, model, nugget, range, sill, seed);
  auto r = esi->estimate(&c_loc, [func](std::string s){
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

  // avoid model construction until we have
  // a more efficient way to pass it through
  //
  // model_list = esi_idw_to_dict(esi);
  model_list = Py_BuildValue("");

  delete esi;

  std::vector<float>().swap(output);
  std::vector<std::vector<float>>().swap(c_smp);
  std::vector<std::vector<float>>().swap(c_loc);
  std::vector<std::vector<float>>().swap(r);
  std::vector<float>().swap(c_val);

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}

static PyObject *loo_esi_kriging_2d(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, has_call, model, seed;
  float alpha, nugget, range, sill;
  std::string fname;
  PyArrayObject *estimation;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ififffiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, & model, &nugget, &range, &sill, &seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==0){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[6] samples should have 2 elements per row (x & y)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[7] scattered should have 2 elements per row (x & y)");
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  // ##### THE METHOD ITSELF #####
  #ifdef DEBUG
  std::cout << "[C++] preparing to call ESI kriging leave-one-out" << "\n";
  #endif

  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  // #ifdef DEBUG
  std::cout << "[C++] creating ESI kriging instance" << "\n";
  // #endif

  sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(c_smp, c_val, lambda, forest_size, bbox, model, nugget, range, sill, seed);

  // #ifdef DEBUG
  std::cout << "[C++] calling ESI kriging leave-one-out" << "\n";
  // #endif
  auto r = esi->leave_one_out([func](std::string s){
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

  // avoid model construction until we have
  // a more efficient way to pass it through
  //
  // model_list = esi_idw_to_dict(esi);
  model_list = Py_BuildValue("");

  delete esi;

  std::vector<float>().swap(output);
  std::vector<std::vector<float>>().swap(c_smp);
  std::vector<std::vector<float>>().swap(c_loc);
  std::vector<std::vector<float>>().swap(r);
  std::vector<float>().swap(c_val);

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}

static PyObject *kfold_esi_kriging_2d(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, has_call, model, k, creation_seed, folding_seed;
  float alpha, nugget, range, sill;
  std::string fname;
  PyArrayObject *estimation;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ififffiiiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &model, &nugget, &range, &sill, &creation_seed, &k, &folding_seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==0){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[6] samples should have 2 elements per row (x & y)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=2){
    PyErr_SetString(PyExc_TypeError, "[7] scattered should have 2 elements per row (x & y)");
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(c_smp, c_val, lambda, forest_size, bbox, model, nugget, range, sill, creation_seed);
  auto r = esi->k_fold(k, [func](std::string s){
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

  // avoid model construction until we have
  // a more efficient way to pass it through
  //
  // model_list = esi_idw_to_dict(esi);
  model_list = Py_BuildValue("");

  delete esi;

  std::vector<float>().swap(output);
  std::vector<std::vector<float>>().swap(c_smp);
  std::vector<std::vector<float>>().swap(c_loc);
  std::vector<std::vector<float>>().swap(r);
  std::vector<float>().swap(c_val);

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}

static PyObject *estimation_esi_kriging_3d(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, has_call, model, seed;
  float alpha, nugget, range, sill;
  std::string fname;
  PyArrayObject *estimation;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ififffiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, & model, &nugget, &range, &sill, &seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==0){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[6] samples should have 3 elements per row (x, y & z)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[7] scattered should have 2 elements per row (x, y & z)");
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(c_smp, c_val, lambda, forest_size, bbox, model, nugget, range, sill, seed);
  auto r = esi->estimate(&c_loc, [func](std::string s){
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

  // avoid model construction until we have
  // a more efficient way to pass it through
  //
  // model_list = esi_idw_to_dict(esi);
  model_list = Py_BuildValue("");

  delete esi;

  std::vector<float>().swap(output);
  std::vector<std::vector<float>>().swap(c_smp);
  std::vector<std::vector<float>>().swap(c_loc);
  std::vector<float>().swap(c_val);

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}

static PyObject *loo_esi_kriging_3d(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, has_call, model, seed;
  float alpha, nugget, range, sill;
  std::string fname;
  PyArrayObject *estimation;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ififffiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, & model, &nugget, &range, &sill, &seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==0){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[6] samples should have 3 elements per row (x, y & z)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[7] scattered should have 2 elements per row (x, y & z)");
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(c_smp, c_val, lambda, forest_size, bbox, model, nugget, range, sill, seed);
  auto r = esi->leave_one_out([func](std::string s){
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

  // avoid model construction until we have
  // a more efficient way to pass it through
  //
  // model_list = esi_idw_to_dict(esi);
  model_list = Py_BuildValue("");

  delete esi;

  std::vector<float>().swap(output);
  std::vector<std::vector<float>>().swap(c_smp);
  std::vector<std::vector<float>>().swap(c_loc);
  std::vector<std::vector<float>>().swap(r);
  std::vector<float>().swap(c_val);

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}

static PyObject *kfold_esi_kriging_3d(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, has_call, model, k, creation_seed, folding_seed;
  float alpha, nugget, range, sill;
  std::string fname;
  PyArrayObject *estimation;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ififffiiiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, & model, &nugget, &range, &sill, &creation_seed, &k, &folding_seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==0){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  c_val = std::vector<float>(smp_sh[0]);
  if (smp_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[6] samples should have 3 elements per row (x, y & z)");
    return (PyObject *) NULL;
  }

  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=3){
    PyErr_SetString(PyExc_TypeError, "[7] scattered should have 2 elements per row (x, y & z)");
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  aux = (float *)PyArray_DATA(values);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(c_smp, c_val, lambda, forest_size, bbox, model, nugget, range, sill, creation_seed);
  auto r = esi->k_fold(k, [func](std::string s){
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

  // avoid model construction until we have
  // a more efficient way to pass it through
  //
  // model_list = esi_idw_to_dict(esi);
  model_list = Py_BuildValue("");

  delete esi;

  std::vector<float>().swap(output);
  std::vector<std::vector<float>>().swap(c_smp);
  std::vector<std::vector<float>>().swap(c_loc);
  std::vector<std::vector<float>>().swap(r);
  std::vector<float>().swap(c_val);

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}

static PyObject *estimation_voronoi_idw(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc, r;
  std::vector<float> c_val;
  int forest_size, has_call, seed;
  float alpha, exp;
  std::string fname;
  PyArrayObject *estimation;

#ifdef DEBUG
  std::cout << "[C++] parsing arguments" << "\n";
#endif
  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!iffiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &exp, &seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return((PyObject *) NULL);
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==0){
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

#ifdef DEBUG
  std::cout << "[C++] checking data format" << "\n";
#endif
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

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
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

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }
  aux = (float *)PyArray_DATA(values);

  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  // ##### THE METHOD ITSELF #####
#ifdef DEBUG
  std::cout << "[C++] arranging parameters" << "\n";
#endif
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }

#ifdef DEBUG
  std::cout << "[C++] building voronoi" << "\n";
#endif
  sptlz::VORONOI_IDW* voronoi = new sptlz::VORONOI_IDW(c_smp, c_val, alpha, forest_size, bbox, exp, seed);

#ifdef DEBUG
  std::cout << "[C++] calling voronoi" << "\n";
#endif
    r = voronoi->estimate(&c_loc, [func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  });

#ifdef DEBUG
  std::cout << "[C++] formatting the output" << "\n";
#endif
  auto output = sptlz::as_1d_array(&r);

  // stuff to return data to python
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

#ifdef DEBUG
  std::cout << "[C++] building the output model" << "\n";
#endif

  // avoid model construction until we have
  // a more efficient way to pass it through
  //
  // model_list = esi_idw_to_dict(esi);
  model_list = Py_BuildValue("");

  delete voronoi;

  std::vector<float>().swap(output);
  std::vector<std::vector<float>>().swap(c_smp);
  std::vector<std::vector<float>>().swap(c_loc);
  std::vector<std::vector<float>>().swap(r);
  std::vector<float>().swap(c_val);

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}

static PyObject *loo_voronoi_idw(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, has_call, seed;
  float alpha, exp;
  std::string fname;
  PyArrayObject *estimation;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!iffiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &exp, &seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==0){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=smp_sh[1]){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have same elements per row as samples");
    return (PyObject *) NULL;
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  aux = (float *)PyArray_DATA(values);
  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }

  sptlz::VORONOI_IDW* voronoi = new sptlz::VORONOI_IDW(c_smp, c_val, alpha, forest_size, bbox, exp, seed);
  auto r = voronoi->leave_one_out([func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  });
  auto output = sptlz::as_1d_array(&r);

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }
  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  // avoid model construction until we have
  // a more efficient way to pass it through
  //
  // model_list = esi_idw_to_dict(esi);
  model_list = Py_BuildValue("");

  delete voronoi;

  std::vector<float>().swap(output);
  std::vector<std::vector<float>>().swap(c_smp);
  std::vector<std::vector<float>>().swap(c_loc);
  std::vector<std::vector<float>>().swap(r);
  std::vector<float>().swap(c_val);

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}

static PyObject *kfold_voronoi_idw(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, has_call, k, creation_seed, folding_seed;
  float alpha, exp;
  std::string fname;
  PyArrayObject *estimation;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!iffiiiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &exp, &creation_seed, &k, &folding_seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==0){
    PyErr_SetString(PyExc_TypeError, "[2] Not callable object");
    return((PyObject *) NULL);
  }
  aux_str = PyObject_GetAttrString(func, "__class__");
  aux_str = PyObject_GetAttrString(aux_str, "__name__");
  fname = PyUnicode_AsUTF8(aux_str);

  // Argument validations
  if (PyArray_NDIM(samples)!=2){
    PyErr_SetString(PyExc_TypeError, "[3] samples must be a 2 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(values)!=1){
    PyErr_SetString(PyExc_TypeError, "[4] values must be a 1 dimensions array");
    return (PyObject *) NULL;
  }
  if (PyArray_NDIM(scattered)!=2){
    PyErr_SetString(PyExc_TypeError, "[5] scattered must be a 2 dimensions array");
    return (PyObject *) NULL;
  }

  npy_intp *smp_sh = PyArray_SHAPE(samples);
  npy_intp *sct_sh = PyArray_SHAPE(scattered);
  if (sct_sh[1]!=smp_sh[1]){
    PyErr_SetString(PyExc_TypeError, "[6] scattered should have same elements per row as samples");
    return (PyObject *) NULL;
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
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
  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  aux = (float *)PyArray_DATA(values);
  c_val = std::vector<float>(smp_sh[0]);
  memcpy(&c_val[0], &aux[0], c_val.size()*sizeof(float));

  // ##### THE METHOD ITSELF #####
  auto bbox = sptlz::samples_coords_bbox(&c_loc);
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }

  sptlz::VORONOI_IDW* voronoi = new sptlz::VORONOI_IDW(c_smp, c_val, alpha, forest_size, bbox, exp, creation_seed);
  auto r = voronoi->k_fold(k, [func](std::string s){
    PyObject *tup = Py_BuildValue("(s)", s.c_str());
    PyObject_Call(func, tup, NULL);
    return(0);
  }, folding_seed);
  auto output = sptlz::as_1d_array(&r);

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  // stuff to return data to python
  const npy_intp dims[2] = {(int)r.size(), forest_size};
  estimation = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
  aux = (float *)PyArray_DATA(estimation);
  memcpy(&aux[0], &output.data()[0], output.size()*sizeof(float));

  // avoid model construction until we have
  // a more efficient way to pass it through
  //
  // model_list = esi_idw_to_dict(esi);
  model_list = Py_BuildValue("");

  delete voronoi;

  std::vector<float>().swap(output);
  std::vector<std::vector<float>>().swap(c_smp);
  std::vector<std::vector<float>>().swap(c_loc);
  std::vector<std::vector<float>>().swap(r);
  std::vector<float>().swap(c_val);

  if (Py_REFCNT(aux) != 0) {
      Py_SET_REFCNT(aux, 0);
  }

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}


static PyMethodDef SpatializeMethods[] = {
  { "get_partitions_using_esi", get_partitions_using_esi, METH_VARARGS, "get several partitions using MondrianTree" },

  // { "estimation_using_model", estimation_using_model, METH_VARARGS, "Using stored model to estimate" },
  // { "loo_using_model", loo_using_model, METH_VARARGS, "Leave-one-out validation using stored model" },
  // { "kfold_using_model", kfold_using_model, METH_VARARGS, "K-fold validation using stored model" },

  { "estimation_nn_idw", estimation_nn_idw, METH_VARARGS, "IDW using nearest neighbors to estimate" },
  { "loo_nn_idw", loo_nn_idw, METH_VARARGS, "Leave-one-out validation for IDW using nearest neighbors" },
  { "kfold_nn_idw", kfold_nn_idw, METH_VARARGS, "K-fold validation for IDW using nearest neighbors" },

  { "estimation_esi_idw", estimation_esi_idw, METH_VARARGS, "IDW using ESI to estimate" },
  { "loo_esi_idw", loo_esi_idw, METH_VARARGS, "Leave-one-out validation for IDW using ESI" },
  { "kfold_esi_idw", kfold_esi_idw, METH_VARARGS, "K-fold validation for IDW using ESI" },

  { "estimation_esi_kriging_2d", estimation_esi_kriging_2d, METH_VARARGS, "Esi using Kriging on 2 dimensions to estimate" },
  { "loo_esi_kriging_2d", loo_esi_kriging_2d, METH_VARARGS, "Leave-one-out validation for Esi using Kriging on 2 dimensions" },
  { "kfold_esi_kriging_2d", kfold_esi_kriging_2d, METH_VARARGS, "K-fold validation for Esi using Kriging on 2 dimensions" },

  { "estimation_esi_kriging_3d", estimation_esi_kriging_3d, METH_VARARGS, "Esi using Kriging on 3 dimensions to estimate" },
  { "loo_esi_kriging_3d", loo_esi_kriging_3d, METH_VARARGS, "Leave-one-out validation for Esi using Kriging on 3 dimensions" },
  { "kfold_esi_kriging_3d", kfold_esi_kriging_3d, METH_VARARGS, "K-fold validation for Esi using Kriging on 3 dimensions" },

  { "estimation_voronoi_idw", estimation_voronoi_idw, METH_VARARGS, "IDW using VORONOI ESI to estimate" },
  { "loo_voronoi_idw", loo_voronoi_idw, METH_VARARGS, "Leave-one-out validation for IDW using VORONOI ESI" },
  { "kfold_voronoi_idw", kfold_voronoi_idw, METH_VARARGS, "K-fold validation for IDW using VORONOI ESI" },

  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef libspatialize = {
    PyModuleDef_HEAD_INIT,
    "libspatialize",   /* name of module */
    "Python wrapper for C++ Ensemble Spatial Interpolation library", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SpatializeMethods
};

PyMODINIT_FUNC PyInit_libspatialize(void){
    PyObject *m = PyModule_Create(&libspatialize);
    if (m == NULL)
        return(NULL);

    // /* Load 'numpy' functionality. */
    import_array();
    return(m);
}
