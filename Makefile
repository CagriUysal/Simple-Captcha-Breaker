CXX=g++
CXXFLAGS=-std=c++11
OPENCV = `pkg-config --cflags --libs opencv`

main:
	g++ -std=c++11 main.cpp `pkg-config --cflags --libs opencv`

test:
	g++ -std=c++11 fourierTest.cpp `pkg-config --cflags --libs opencv`
	
dunno:
	$(CXX) -$(CXXFLAGS)  -o $@.cpp $(OPENCV)
