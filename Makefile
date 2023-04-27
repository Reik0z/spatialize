all: setup.py
	python3 setup.py build_ext --inplace

clean:
	rm -Rf build* *.so


test: all
	python3 test/test_spatialize.py
