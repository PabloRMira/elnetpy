mytest:
	g++ test.cpp -o test
	./test

utest:
	coverage run -m pytest tests/unit_tests
	coverage report

ptest:
	pytest tests/performance_tests

pycov:
	coverage run -m pytest tests/unit_tests
	coverage-badge -f -o img/coverage.svg
