CXX=g++
CXXFLAGS=-std=c++20

llm: bin/llm

bin/llm: ./ngram_model.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $$(pkg-config --cflags --libs arrow parquet) -Iincludes