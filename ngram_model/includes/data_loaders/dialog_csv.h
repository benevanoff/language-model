#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

std::vector<std::string> parseCSVLine(const std::string &line);

std::pair<std::string, std::string> parseUtterances(const std::string &line);