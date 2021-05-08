mytest:
	g++ test.cpp -o test
	./test

utest:
	coverage run -m pytest tests/unit_tests
	coverage report

ptest:
	pytest tests/performance_tests

prepush: pycov

pycov:
	coverage run -m pytest tests/unit_tests
	coverage report
	coverage-badge -f -o img/coverage.svg

upmaster:
	git checkout master
	git pull origin master
