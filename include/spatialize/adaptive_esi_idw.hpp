/*
========================================================================================================================
static PyObject *loo_adaptive_esi_idw_2d(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, has_call, seed;
  float alpha;
  std::string fname;
  PyArrayObject *estimation;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ifiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==NULL){
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
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ADAPTIVE_ESI_IDW* esi = new sptlz::ADAPTIVE_ESI_IDW(c_smp, c_val, lambda, forest_size, bbox, seed);
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

  model_list = esi_idw_anis_to_dict(esi);

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}

static PyObject *kfold_adaptive_esi_idw_2d(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, has_call, k, creation_seed, folding_seed;
  float alpha;
  std::string fname;
  PyArrayObject *estimation;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ifiiiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &creation_seed, &k, &folding_seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==NULL){
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
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ADAPTIVE_ESI_IDW* esi = new sptlz::ADAPTIVE_ESI_IDW(c_smp, c_val, lambda, forest_size, bbox, creation_seed);
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

  model_list = esi_idw_anis_to_dict(esi);

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}

static PyObject *estimation_adaptive_esi_idw_3d(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, has_call, seed;
  float alpha;
  std::string fname;
  PyArrayObject *estimation;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ifiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==NULL){
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
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ADAPTIVE_ESI_IDW* esi = new sptlz::ADAPTIVE_ESI_IDW(c_smp, c_val, lambda, forest_size, bbox, seed);
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

  model_list = esi_idw_anis_to_dict(esi);

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}

static PyObject *loo_adaptive_esi_idw_3d(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, has_call, seed;
  float alpha;
  std::string fname;
  PyArrayObject *estimation;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ifiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==NULL){
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
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ADAPTIVE_ESI_IDW* esi = new sptlz::ADAPTIVE_ESI_IDW(c_smp, c_val, lambda, forest_size, bbox, seed);
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

  model_list = esi_idw_anis_to_dict(esi);

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}

static PyObject *kfold_adaptive_esi_idw_3d(PyObject *self, PyObject *args){
  PyObject *func, *aux_str, *model_list;
  PyArrayObject *samples, *values, *scattered;
  float *aux;
  std::vector<std::vector<float>> c_smp, c_loc;
  std::vector<float> c_val;
  int forest_size, has_call, k, creation_seed, folding_seed;
  float alpha;
  std::string fname;
  PyArrayObject *estimation;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!ifiiiO!O", &PyArray_Type, &samples, &PyArray_Type, &values, &forest_size, &alpha, &creation_seed, &k, &folding_seed, &PyArray_Type, &scattered, &func)) {
    PyErr_SetString(PyExc_TypeError, "[1] Argument do not match");
    return (PyObject *) NULL;
  }

  has_call = PyObject_HasAttrString(func, "__call__");
  if(has_call==NULL){
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
  auto bbox2 = sptlz::samples_coords_bbox(&c_smp);
  for(int i=0;i<smp_sh[1];i++){
    if(bbox2.at(i).at(0) < bbox.at(i).at(0)){bbox.at(i).at(0) = bbox2.at(i).at(0);}
    if(bbox2.at(i).at(1) > bbox.at(i).at(1)){bbox.at(i).at(1) = bbox2.at(i).at(1);}
  }
  float lambda = sptlz::bbox_sum_interval(bbox);
  lambda = 1/(lambda-alpha*lambda);

  sptlz::ADAPTIVE_ESI_IDW* esi = new sptlz::ADAPTIVE_ESI_IDW(c_smp, c_val, lambda, forest_size, bbox, creation_seed);
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

  model_list = esi_idw_anis_to_dict(esi);

  return(Py_BuildValue("O,O", model_list, (PyObject *)estimation));
}

========================================================================================================================
  { "estimation_adaptive_esi_idw_2d", estimation_adaptive_esi_idw_2d, METH_VARARGS, "Adaptive Esi using IDW on 2 dimensions to estimate" },
  { "loo_adaptive_esi_idw_2d", loo_adaptive_esi_idw_2d, METH_VARARGS, "Leave-one-out validation for Adaptive Esi using IDW on 2 dimensions" },
  { "kfold_adaptive_esi_idw_2d", kfold_adaptive_esi_idw_2d, METH_VARARGS, "K-fold validation for Adaptive Esi using IDW on 2 dimensions" },

  { "estimation_adaptive_esi_idw_3d", estimation_adaptive_esi_idw_3d, METH_VARARGS, "Adaptive Esi using IDW on 3 dimensions to estimate" },
  { "loo_adaptive_esi_idw_3d", loo_adaptive_esi_idw_3d, METH_VARARGS, "Leave-one-out validation for Adaptive Esi using IDW on 3 dimensions" },
  { "kfold_adaptive_esi_idw_3d", kfold_adaptive_esi_idw_3d, METH_VARARGS, "K-fold validation for Adaptive Esi using IDW on 3 dimensions" },
*/

#ifndef _SPTLZ_ADTV_ESI_IDW_
#define _SPTLZ_ADTV_ESI_IDW_

#include <stdexcept>
#include <cmath>
#include <random>
#include "spatialize/abstract_esi.hpp"
#include "spatialize/utils.hpp"

namespace sptlz{
  class LOO2D{
    protected:
      std::vector<std::vector<float>> *coords;
      std::vector<float> *coords1d;
      std::vector<float> *values;
      std::vector<float> centroid;

    public:
      LOO2D(std::vector<std::vector<float>> *_coords, std::vector<float> *_values){
        values = _values;
        centroid = sptlz::get_centroid(_coords);
        coords = _coords;
        auto aux = sptlz::as_1d_array(_coords);
        coords1d = &aux;
      }

      float eval(std::vector<float> X){
        int n = values->size();
        float r = 0.0, sum_w, est, wj;
        std::vector<float> params = {X.at(1), X.at(2)};
        auto tr_coords = sptlz::transform(coords, &params, &centroid);

        for(int i=0; i<n; i++){
          auto ds = sptlz::distances(&tr_coords, i);
          sum_w = 0.0;
          est = 0.0;
          for(int j=0;j<n;j++){
            if(j!=i){
              wj = 1.0/(1.0+std::pow(ds.at(j), X.at(0)));
              sum_w += wj;
              est += wj*values->at(j);
            }
          }
          r += std::pow(values->at(i)-est/sum_w, 2.0);
        }
        return(r/n);
      }
  };

  class LOO3D{
    protected:
      std::vector<std::vector<float>> *coords;
      std::vector<float> *coords1d;
      std::vector<float> *values;
      std::vector<float> centroid;

    public:
      LOO3D(std::vector<std::vector<float>> *_coords, std::vector<float> *_values){
        values = _values;
        centroid = sptlz::get_centroid(_coords);
        coords = _coords;
        auto aux = sptlz::as_1d_array(_coords);
        coords1d = &aux;
      }

      float eval(std::vector<float> X){
        int n = values->size();
        float r = 0.0, sum_w, est, wj;
        std::vector<float> params = {X.at(1), X.at(2), X.at(3), X.at(4), X.at(5)};
        auto tr_coords = sptlz::transform(coords, &params, &centroid);

        for(int i=0; i<n; i++){
          auto ds = sptlz::distances(&tr_coords, i);
          sum_w = 0.0;
          est = 0.0;
          for(int j=0;j<n;j++){
            if(j!=i){
              wj = 1.0/(1.0+std::pow(ds.at(j), X.at(0)));
              sum_w += wj;
              est += wj*values->at(j);
            }
          }
          r += std::pow(values->at(i)-est/sum_w, 2.0);
        }
        return(r/n);
      }
  };

  class ADAPTIVE_ESI_IDW: public ESI {
    protected:
      std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params){
        std::vector<float> result;

        std::cout << "sample id size = "<< samples_id->size() << std::endl;
        std::cout << "params size = "<< params->size() << std::endl;

        if(samples_id->size()==0){
          for([[maybe_unused]] auto l: *locations_id){
            result.push_back(NAN);
          }
          return(result);
        }
        std::cout << "1" << std::endl;

        if(samples_id->size()==1){
          for([[maybe_unused]] auto l: *locations_id){
            result.push_back(params->at(0));
            //result.push_back(values->at(samples_id->at(0)));
          }
          return(result);
        }
        std::cout << "2" << std::endl;

        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        auto sl_locations = slice(locations, locations_id);
        std::vector<float> centroid;
        int i_params = 0;

        centroid.push_back(params->at(i_params++));
        centroid.push_back(params->at(i_params++));

        if(coords->at(0).size()==3){
          centroid.push_back(params->at(i_params++));
        }
        std::cout << "3" << std::endl;

        float exponent = params->at(i_params++);
        std::vector<float> rot_params = slice_from(params, i_params);

        std::cout << "4" << std::endl;
        std::cout << "|sl_coords|" << sl_coords.size() << std::endl;
        std::cout << "|rot_params|" << rot_params.size() << std::endl;
        std::cout << "rot_params";
        pprint(rot_params);
        std::cout << "|centroid|" << centroid.size() << std::endl;
        std::cout << "centroid";
        pprint(centroid);

        auto tr_coords = transform(&sl_coords, &rot_params, &centroid);
        std::cout << "what";
        auto tr_locations = transform(&sl_locations, &rot_params, &centroid);

        std::cout << "5" << std::endl;

        float w, w_sum, w_v_sum;

        // for every location
        for(int i=0; i<locations_id->size(); i++){
          w_sum = 0.0;
          w_v_sum = 0.0;

          for(int j=0; j<samples_id->size(); j++){
            // calculate weight
            w = 1/(1+std::pow(distance(&(tr_locations.at(i)), &(tr_coords.at(j))), exponent));
            // keep sum of weighted values and sum of weights
            w_sum += w;
            w_v_sum += w*sl_values.at(j);
          }
          // return weighted values sum normalized (divided by weights sum)
          result.push_back(w_v_sum/w_sum);
        }
        std::cout << "|result|: " << result.size();

        return(result);
      }

      std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params){
        std::vector<float> result;

        if((samples_id->size()==0) || (samples_id->size()==1)){
          for([[maybe_unused]] auto l: *samples_id){
            result.push_back(NAN);
          }
          return(result);
        }

        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        std::vector<float> centroid;
        int i_params = 0;
        centroid.push_back(params->at(i_params++));
        centroid.push_back(params->at(i_params++));
        if(coords->at(0).size()==3){
          centroid.push_back(params->at(i_params++));
        }

        float exponent = params->at(i_params++);
        std::vector<float> rot_params = slice_from(params, i_params);

        auto tr_coords = transform(&sl_coords, &rot_params, &centroid);

        float w, w_sum, w_v_sum;

        // for every location
        for(int i=0; i<samples_id->size(); i++){
          w_sum = 0.0;
          w_v_sum = 0.0;

          for(int j=0; j<samples_id->size(); j++){
            if(i!=j){
              // calculate weight
              w = 1/(1+std::pow(distance(&(tr_coords.at(i)), &(tr_coords.at(j))), exponent));
              // keep sum of weighted values and sum of weights
              w_sum += w;
              w_v_sum += w*sl_values.at(j);
            }
          }
          // return weighted values sum normalized (divided by weights sum)
          result.push_back(w_v_sum/w_sum);
        }
        return(result);
      }

      std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *folds, std::vector<int> *samples_id, std::vector<float> *params){
        std::vector<float> result(samples_id->size());
        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        auto sl_folds = slice(folds, samples_id);
        float w, w_sum, w_v_sum;

        if((samples_id->size()==0) || (samples_id->size()==1)){
          for([[maybe_unused]] auto l: *samples_id){
            result.push_back(NAN);
          }
          return(result);
        }

        std::vector<float> centroid;
        int i_params = 0;
        centroid.push_back(params->at(i_params++));
        centroid.push_back(params->at(i_params++));
        if(coords->at(0).size()==3){
          centroid.push_back(params->at(i_params++));
        }

        float exponent = params->at(i_params++);
        std::vector<float> rot_params = slice_from(params, i_params);
        auto tr_coords = transform(&sl_coords, &rot_params, &centroid);

        for(int i=0; i<k; i++){
          auto test_train = indexes_by_predicate<int>(&sl_folds, [i](int *j){return(*j==i);});
          if(test_train.first.size()!=0){ // if is 0, then there's nothing to estimate
            if(test_train.second.size()==0){
              for(int j: test_train.first){
                result.at(j) = NAN;
              }
            }else{
              for(int j: test_train.first){
                w_sum = 0.0;
                w_v_sum = 0.0;
                for(int l: test_train.second){
                  w = 1/(1+std::pow(distance(&(tr_coords.at(j)), &(tr_coords.at(l))), exponent));
                  w_sum += w;
                  w_v_sum += w*values->at(samples_id->at(l));
                }
                result.at(j) = w_v_sum/w_sum;
              }
            }
          }
        }
        return(result);
      }

      void post_process(){
        std::vector<std::vector<float>> leaf_coords;
        std::vector<float> leaf_values;

        for(int i=0; i<mondrian_forest.size(); i++){
          auto mt = mondrian_forest.at(i);
          for(int j=0; j<mt->samples_by_leaf.size(); j++){
            leaf_coords.clear();
            leaf_values.clear();
            for(int k=0; k<mt->samples_by_leaf.at(j).size(); k++){
              leaf_coords.push_back(coords.at(mt->samples_by_leaf.at(j).at(k)));
              leaf_values.push_back(values.at(mt->samples_by_leaf.at(j).at(k)));
            }/*
            std::cout << "|iforest|: " << i << " |jleaf|: " << j << std::endl;
            std::cout << "|coords|: " << leaf_coords.size() << std::endl;
            std::cout << "|values|: " << leaf_values.size() << std::endl;
            auto bla = get_params(&leaf_coords, &leaf_values);
            std::cout << "|params|: " << bla.size() << std::endl;
            mt->leaf_params.at(j) = bla;*/
            mt->leaf_params.at(j) = get_params(&leaf_coords, &leaf_values);
          }
          std::cout << "pp: " << i + 1 << "/" << mondrian_forest.size() << std::endl;
        }
      }

      std::vector<float> get_params(std::vector<std::vector<float>> *coords, std::vector<float> *values){
        std::uniform_real_distribution<float> uni_float(0, 1);
        int best_of = 3;
        if(coords->size()==0){
          return(std::vector<float>());
        }else if(coords->size()==1){
          return(std::vector<float>({values->at(0)}));
        }

        std::vector<float> min_coords;
        if(coords->at(0).size()==2){
          std::vector<float> starting_point, candidate;
          float min_value=1e20, aux;
          LOO2D *func = new LOO2D(coords, values);
          std::vector<std::vector<float>> ranges = {
            {0.1, 8.0, 0.2}, // exp
            {0.0, 180.0, 1.0}, // azim
            {0.1, 1.001, 0.05} // ratio
          };
          for(int i=0; i<best_of; i++){
            starting_point = {};
            for(int j=0; j<ranges.size(); j++){
              starting_point.push_back(ranges.at(j).at(0)+uni_float(my_rand)*(ranges.at(j).at(1)-ranges.at(j).at(0)));
            }
            candidate = sptlz::grid_search<LOO2D>(func, &ranges, starting_point);
            aux = func->eval(candidate);
            if(aux<min_value){
              min_coords = candidate;
              min_value = aux;
            }
          }
        }else if(coords->at(0).size()==3){
          std::vector<float> starting_point, candidate;
          float min_value=1e20, aux;
          LOO3D *func = new LOO3D(coords, values);
          std::vector<std::vector<float>> ranges = {
            {0.1, 8.0, 0.2}, // exp
            {0.0, 180.0, 1.0}, // azim
            {0.0, 180.0, 1.0}, // dip
            {0.0, 180.0, 1.0}, // plunge
            {0.1, 1.001, 0.05}, // ratio1
            {0.1, 1.001, 0.05} // ratio2
          };
          for(int i=0; i<best_of; i++){
            starting_point = {};
            for(int j=0; j<ranges.size(); j++){
              starting_point.push_back(ranges.at(j).at(0)+uni_float(my_rand)*(ranges.at(j).at(1)-ranges.at(j).at(0)));
            }
            candidate = sptlz::grid_search<LOO3D>(func, &ranges, starting_point);
            aux = func->eval(candidate);
            if(aux<min_value){
              min_coords = candidate;
              min_value = aux;
            }
          }
        }

        if(min_coords.size()==0){ // don't know why sometimes it can't get a candidate
          if (coords->at(0).size()==2){
            min_coords = {2.0, 0.0, 1.0};
          }else if(coords->at(0).size()==3){
            min_coords = {2.0, 0.0, 0.0, 0.0, 1.0, 1.0};
          }
          //return(std::vector<float>());
        }
        auto centroid = sptlz::get_centroid(coords);
        for(auto v: min_coords){
          centroid.push_back(v);
        }

        return(centroid);
      }

    public:
      ADAPTIVE_ESI_IDW(std::vector<std::vector<float>> _coords, std::vector<float> _values, float lambda, int forest_size, std::vector<std::vector<float>> bbox, int seed=206936):ESI(_coords, _values, lambda, forest_size, bbox, seed){
        post_process();
      }

      ADAPTIVE_ESI_IDW(std::vector<sptlz::MondrianTree*> _mondrian_forest, std::vector<std::vector<float>> _coords, std::vector<float> _values):ESI(_mondrian_forest, _coords, _values){}
  };
}

#endif
