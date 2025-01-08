# Spatialize: A Python wrapper for C++ ESI library

## What is it?

Spatialize is an open source library that implements _ensemble spatial interpolation_, 
a novel method that combines the simplicity of basic interpolation methods with 
the power of classical geoestatistical tools, like Kriging.

This library aims to bridge the gap between expert and non-expert users of geostatistics 
by providing automated tools that rival traditional geostatistical methods.


Main features of the library include:

- Stochastic modelling and ensemble learning, making it robust, scalable and suitable for large datasets.
- Provides a powerful framework for uncertainty quantification, offering both point estimates and empirical posterior distributions.
- It is implemented in Python 3.x, with a C++ core for improved performance.
- It is designed to be easy to use, requiring minimal user intervention. 

## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/alges/spatialize

Direct installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/spatialize).

### PyPI
```bash
pip install spatialize
```

## Dependencies
- [NumPy: Powerful n-dimensional arrays and numerical computing tools](https://www.numpy.org)
- [pandas: Fast, powerful, flexible and easy to use open source data analysis and manipulation tool](https://pandas.pydata.org)
- [Matplotlib: Visualization with Python](https://matplotlib.org/)
- [scikit-learn: Machine Learning in Python](https://scikit-learn.org/)
- [SciPy: Fundamental algorithms for scientific computing in Python](https://scipy.org/)

## License
[Apache-2.0](LICENSE)

## Acknowledge
Please cite the following paper when publishing work relating to this library:
  
    @article{spatialize2025,
        title = {Spatialize: A Python/C++ Library for Ensemble Spatial Interpolation},
	    author = {Ega{\~n}a, {\'A}lvaro F. and Ehrenfeld, Alejandro and Navarro, Felipe and Garrido, Felipe and Valenzuela, Mar{\'i}a Jes{\'u}s and S{\'a}nchez-P{\'e}rez, Juan F. },
	    date = {},
	    doi = {},
	    isbn = {},
	    journal = {},
	    number = {},
	    pages = {},
	    url = {},
	    volume = {},
	    year = {2025},
     }
