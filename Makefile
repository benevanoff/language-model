CXX=g++
CXXFLAGS=-std=c++20

llm: bin/llm

bin/llm: ./bigram_model.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@