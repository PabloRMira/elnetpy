mytest:
	g++ test.cpp -o test
	./test

upytest:
	pytest tests/unit_tests
