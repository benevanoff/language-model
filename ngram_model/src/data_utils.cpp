#include "data_utils.h"

void replace_commas_with_spaces(std::string& str) {
    size_t pos = 0;
    while ((pos = str.find(',', pos)) != std::string::npos) {
        str.replace(pos, 1, " ");
        pos += 1; 
    }
}

bool is_trailing_punct(char c) {
    return c == '.' || c == '!' || c == '?';
}

std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream stream(line);
    std::string token;
    
    while (stream >> token) {
        size_t len = token.length();
        size_t split_pos = len;
        
        // find the position where trailing punctuation starts
        while (split_pos > 0 && is_trailing_punct(token[split_pos - 1])) {
            split_pos--;
        }
        
        if (split_pos == 0) {
            // the entire token is punctuation (e.g., "!!")
            tokens.push_back(token);
        }
        else if (split_pos < len) {
            // split the token into word and punctuation
            std::string word = token.substr(0, split_pos);
            std::string punct = token.substr(split_pos);
            tokens.push_back(word);
            tokens.push_back(punct);
        }
        else {
            // no trailing punctuation
            tokens.push_back(token);
        }
    }
    
    return tokens;
}

int filter_vocab(int n, std::unordered_map<std::string, std::unordered_map<std::string, int>> &vocab_dist) {
    // filter out all tokens which have been seen less than N times
    // return the number of unique ngrams left after the filtering
    int unique_ngrams = 0;
    for(auto outer_it = vocab_dist.begin(); outer_it != vocab_dist.end(); ) {
        auto &inner_map = outer_it->second;
        for(auto inner_it = inner_map.begin(); inner_it != inner_map.end(); ) {
            if(inner_it->second < n) {
                inner_it = inner_map.erase(inner_it);
            }
            else {
                inner_it++;
                unique_ngrams++;
            }
        }
        if(inner_map.empty())
            outer_it = vocab_dist.erase(outer_it);
        else
            outer_it++;
    }
    std::cout << "unique n grams: " << unique_ngrams << std::endl;
    return unique_ngrams;
}

void weights_to_file(std::string filename, const std::unordered_map<std::string, std::unordered_map<std::string, int>> &vocab_dist) {
    std::ofstream ofs;
    ofs.open(filename);
    for (auto it = vocab_dist.begin(); it != vocab_dist.end(); it++) {
        ofs << it->first << std::endl;
        for (auto jt = it->second.begin(); jt != it->second.end(); jt++)
            ofs << jt->first << " " << jt->second << std::endl;
        ofs << std::endl;
    }
}