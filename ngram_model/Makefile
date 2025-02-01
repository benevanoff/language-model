CXX=g++
CXXFLAGS=-std=c++23

llm: bin/llm

bin/llm: ./main.cpp
	$(CXX) $(CXXFLAGS) -o $@ src/ngram_model.cpp src/data_loaders.cpp src/data_utils.cpp src/dialog_csv.cpp  $< $$(pkg-config --cflags --libs arrow parquet) -Iincludes

tests: bin/tests

bin/tests: ./test_ngram.cpp
	$(CXX) $(CXXFLAGS) -o $@ src/ngram_model.cpp src/data_loaders.cpp src/data_utils.cpp src/dialog_csv.cpp $< $$(pkg-config --cflags --libs arrow parquet) -Iincludes