#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <exception>
#include <unordered_map>
#include <sstream>

std::vector<std::string> splitByWhitespace(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream stream(line);
    std::string token;
    
    while (stream >> token) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::string parse_utterances_from_dialog_file(std::ifstream *file) {
    std::string line;
    std::getline(*file, line);
    return line.substr(line.find(":")+2);
}

std::unordered_map<std::string, std::unordered_map<std::string, int>> calc_word_dist() {
    std::ifstream ifs;
    ifs.open("data/combined.csv");

    std::string line;
    std::getline(ifs, line); // discard head


    std::unordered_map<std::string, std::unordered_map<std::string, int>> vocab_dist;

    while (ifs.good()) {    
        try {
            std::string utt = parse_utterances_from_dialog_file(&ifs);
            // std::cout << utt << std::endl;
            std::vector<std::string> tokens = splitByWhitespace(utt);

            tokens.insert(tokens.begin(), "<start>");
            for (size_t i = 1; i < tokens.size(); i++) {
                // std::cout << "(" << tokens[i-1] << ") " << tokens[i] << std::endl;
                std::string ngram = tokens[i-1];
                // std::cout << "ngram " << ngram << std::endl;
                if (!vocab_dist.contains(ngram)) {
                    std::unordered_map<std::string, int> tmp;
                    tmp[tokens[i]] = 1;
                    vocab_dist[ngram] = tmp;
                } else {
                    vocab_dist[ngram][tokens[i]] += 1;
                }
            }
        } catch (std::exception e) {}
    }


    std::unordered_map<std::string, std::unordered_map<std::string, int>> vocab_dist_filtered;
    for (auto it = vocab_dist.begin(); it != vocab_dist.end(); it++) {
        std::unordered_map<std::string, int> inner;
        for (auto itt = it->second.begin(); itt != it->second.end(); itt++) {
            if (itt->second > 1) {
                inner[itt->first] = itt->second;
            }
            if (inner.size() > 0)
                vocab_dist_filtered[it->first] = inner;
        }
    }
    
    int unique_ngrams = 0;
    for (auto it = vocab_dist_filtered.begin(); it != vocab_dist_filtered.end(); it++) {
        unique_ngrams++;
        std::cout << "----------" << std::endl;
        std::cout << it->first << std::endl;
        for (auto itt = it->second.begin(); itt != it->second.end(); itt++) {
            std::cout << itt->first << ": " << itt->second << std::endl;
        }
        std::cout << "----------" << std::endl;
    }

    std::cout << "unique_ngrams " << unique_ngrams << std::endl;
    return vocab_dist_filtered;
}

std::string predict_next_word(std::unordered_map<std::string, std::unordered_map<std::string, int>> vocab_dist, std::string prev_word) {
    // std::cout << "predicting next word for " << prev_word << std::endl;
    std::vector<std::string> dist_space;
    try {
        for (auto it = vocab_dist.at(prev_word).begin(); it != vocab_dist.at(prev_word).end(); it++) {
            for (int i = 0; i < it->second; i++) dist_space.push_back(it->first);
        }
        return dist_space.at(rand() % dist_space.size());
    } catch (std::exception e) {
        auto it = vocab_dist.begin();
        std::advance(it, rand()%vocab_dist.size());
        return it->first;
    }
}

int main() {
    std::unordered_map<std::string, std::unordered_map<std::string, int>> vocab_dist = calc_word_dist();

    std::cout << "Starting word: ";
    std::string next_word;
    std::getline(std::cin, next_word);
    while (true) {
        next_word = predict_next_word(vocab_dist, next_word);
        std::cout << "prediction: " << next_word;
        // wait for input from console to continue
        std::string dummy;
        std::getline(std::cin, dummy);
    }
    return 0;
}