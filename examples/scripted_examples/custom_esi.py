import time, numpy as np, pandas as pd, libspatialize as lsp
from numba import njit

def perc(s):
	print('Processing... {0}'.format(s), flush=True)

@njit
def _leaf_estimation(smp: np.ndarray, val: np.ndarray, qry: np.ndarray, params: np.ndarray) -> np.ndarray:
	result = []
	for q in qry:
		dists = np.array([np.sqrt(np.sum(np.power(s-q, 2.0*np.ones((smp.shape[1],))))) for s in smp])
		weights = 1.0 / (1 + np.power(dists, 2.0*np.ones((len(dists),))))
		norm = np.sum(weights)
		result.append(np.sum(val*weights)/norm)
	return(np.array(result))

def leaf_estimation(smp: np.ndarray, val: np.ndarray, qry: np.ndarray, params: np.ndarray) -> np.ndarray:
	result = []
	_sum = np.sum
	_sq = np.sqrt
	_arr = np.array
	#for q in qry:
	def bla(q):
		dist = np.vectorize(lambda x: _sq(_sum((x-q)**2)), signature='(n)->()')
		dists = dist(smp)
		weights = 1.0 / (1 + dists**2)
		norm = _sum(weights)
		return(_sum(val*weights)/norm)
	est = np.vectorize(bla, signature='(n)->()', otypes=[float])
	result = est(qry)
	return(_arr(result))

with open('/home/fgarrido/Dropbox/alges-dev/spatialize/data/muestras.dat', 'r') as data:
	lines = data.readlines()
	lines = [l.strip().split() for l in lines[8:]]
	aux = np.float32(lines)
samples = pd.DataFrame(aux, columns=['X','Y','Z','cu','au','rocktype'])
queries = pd.read_csv('/home/fgarrido/Dropbox/alges-dev/spatialize/data/box.csv')

t = time.time()
est1 = lsp.estimation_esi_idw(
	samples[['X', 'Y', 'Z']].values, 
	samples[['cu']].values[:,0],
	100, 0.7, 2.0, 206936,
	queries[['X', 'Y', 'Z']].values,
	perc
)
t1 = time.time()-t

t = time.time()
est2 = lsp.estimation_custom_esi(
	samples[['X', 'Y', 'Z']].values, 
	samples[['cu']].values[:,0],
	100, 0.7, 206936,
	queries[['X', 'Y', 'Z']].values,
	None, 
	_leaf_estimation,
	perc
)
t2 = time.time()-t

r = sorted(list((est1[1]-est2[1]).flatten()))
print('%d:%f'%(t1//60, t1%60), '%d:%f'%(t2//60, t2%60), r[-1])
