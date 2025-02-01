#pragma once

#include "data_utils.h"
#include "data_loaders/data_loaders.h"

std::unordered_map<std::string, std::unordered_map<std::string, int>> calc_word_dist(int n);

std::string predict_next_word(std::unordered_map<std::string, std::unordered_map<std::string, int>> vocab_dist, std::string prev_word);

std::string predict_next_sentence(std::unordered_map<std::string, std::unordered_map<std::string, int>> vocab_dist, std::vector<std::string> history, int n);