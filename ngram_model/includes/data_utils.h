#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <exception>
#include <unordered_map>
#include <sstream>

void replace_commas_with_spaces(std::string& str);

bool is_trailing_punct(char c);

std::vector<std::string> tokenize(const std::string& line);

int filter_vocab(int n, std::unordered_map<std::string, std::unordered_map<std::string, int>> &vocab_dist);

void weights_to_file(std::string filename, const std::unordered_map<std::string, std::unordered_map<std::string, int>> &vocab_dist);