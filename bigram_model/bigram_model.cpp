#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <exception>
#include <unordered_map>
#include <sstream>

std::vector<std::string> tokenize(const std::string& line) {
    // naive whitespace tokenization
    std::vector<std::string> tokens;
    std::stringstream stream(line);
    std::string token;
    while (stream >> token)
        tokens.push_back(token);
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

std::unordered_map<std::string, std::unordered_map<std::string, int>> calc_word_dist() {
    // count up how many times each word occurs after each ngram in the training set to build a probability distribution
    // example:
    // utterance 1 - "the dog gave the cat a bird"
    // utterance 2 - "the cat caught a bird"
    // {
    //   "<start|start>": {"the": 2}, "<start>|the": {"dog": 1, "cat": 1}, "the|dog": {"gave": 1}, "the|cat": {"caught": 1, "a": 1},
    //   "cat|caugt": {"a": 1}, "dog|gave": {"the": 1}, "cat|a": {"bird": 1}, "caught|a": {"bird": 1} "a|bird": {"<end>": 2}
    // }
    std::cout << "crunching numbers..." << std::endl;
    
    // track the start time so we can measure how long training takes
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    std::ifstream ifs;
    ifs.open("../data/combined.csv");

    std::string line;
    std::getline(ifs, line); // discard head

    std::unordered_map<std::string, std::unordered_map<std::string, int>> vocab_dist;

    while (ifs.good()) {    
        try {
            // read each line from the data file
            // each line should contain a single utterance from a single interlocuter
            std::getline(ifs, line);
            std::string utt = line.substr(line.find(":")+2);
            std::vector<std::string> tokens = tokenize(utt);

            // add start and stop tokens in utterance preprocessing
            tokens.insert(tokens.begin(), "<start>");
            tokens.insert(tokens.begin(), "<start>");
            tokens.push_back("<stop>");

            // loop through the tokens in the utterance and count up how many times each word comes after each ngram
            for (size_t i = 2; i < tokens.size(); i++) {
                std::string ngram = tokens[i-2] + "|" + tokens[i-1];
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

    // filter out all ngrams seen less than 3 times in the training set
    filter_vocab(3, vocab_dist);

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    std::cout << "training time: " << cpu_time_used << "s" << std::endl;
    
    return vocab_dist;
}

std::string predict_next_word(std::unordered_map<std::string, std::unordered_map<std::string, int>> vocab_dist, std::string prev_word) {
    try {
        // fill up a vector of word tickets which mirrors the word distribution
        // if our vocab looks like {"a": 3, "the": 2, "apple": 1, "bird": 1} then our weighted vector should look like [a,a,a,the,the,apple,bird]
        std::vector<std::string> dist_space;
        for (auto it = vocab_dist.at(prev_word).begin(); it != vocab_dist.at(prev_word).end(); it++) {
            for (int i = 0; i < it->second; i++) dist_space.push_back(it->first);
        }
        // choose a random ticket from our "lottery hat" which is weighted to our discrete word distribution
        return dist_space.at(rand() % dist_space.size());
    } catch (std::exception e) {
        // if the prior isnt in our vocab - pick a random word from the vocab
        auto it = vocab_dist.begin();
        std::advance(it, rand()%vocab_dist.size());
        return (it->first).substr(0, it->first.find("|"));
    }
}

int main() {
    srand(time(NULL));
    // calculate the word distribution from the training set - aka unsupervised learning
    std::unordered_map<std::string, std::unordered_map<std::string, int>> vocab_dist = calc_word_dist();
    
    // initialize sentence with start token and a user provided word
    std::string w_minus_two = "<start>";
    std::string w_minus_one;
    // prompt the user for an initial word to kick start the sentence generation
    std::cout << "Starting word: ";
    std::getline(std::cin, w_minus_one);
    
    while (true) {
        // predict the next word and print it to the console
        std::string next_word = predict_next_word(vocab_dist, w_minus_two + "|" + w_minus_one);
        std::cout << "prediction: " << next_word;
        // wait for any input from console to continue
        std::string dummy;
        std::getline(std::cin, dummy);
        // advance the context window, adding the predicted word to the history (aka auto regressive generation)
        w_minus_two = w_minus_one;
        w_minus_one = next_word;
    }
    return 0;
}