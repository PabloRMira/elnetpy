mytest:
	g++ test.cpp -o test
	./test

utest:
	pytest tests/unit_tests

ptest:
	pytest tests/performance_tests

pycov:
	coverage run -m pytest tests/unit_tests
	coverage-badge -o img/coverage.svg
