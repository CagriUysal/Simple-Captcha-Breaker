compile:
	g++ -std=c++11 divDigits.cpp -o breaker `pkg-config --cflags --libs opencv` -g 
	
