mytest:
	g++ test.cpp -o test
	./test

upytest:
	pytest tests/unit_tests

pycov:
	coverage run -m pytest tests/unit_tests
	coverage-badge -o img/coverage.svg
