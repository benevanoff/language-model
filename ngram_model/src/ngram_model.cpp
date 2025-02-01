#include "data_utils.h"
#include "data_loaders/data_loaders.h"
#include "ngram_model.h"

std::unordered_map<std::string, std::unordered_map<std::string, int>> calc_word_dist(int n) {
    // count up how many times each word occurs after each ngram in the training set to build a probability distribution
    // example:
    // utterance 1 - "the dog gave the cat a bird"
    // utterance 2 - "the cat caught a bird"
    // {
    //   "<start|start>": {"the": 2}, "<start>|the": {"dog": 1, "cat": 1}, "the|dog": {"gave": 1}, "the|cat": {"caught": 1, "a": 1},
    //   "cat|caugt": {"a": 1}, "dog|gave": {"the": 1}, "cat|a": {"bird": 1}, "caught|a": {"bird": 1} "a|bird": {"<end>": 2}
    // }
    std::cout << "crunching numbers...." << std::endl;
    
    // track the start time so we can measure how long training takes
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    // load the datasets
    std::vector<std::string> utterances = parse_utterances_from_soda_parquet();
    std::cout << "finished collecting soda utterances " << utterances.size() << std::endl;
    //std::vector<std::string> utterances2 = parse_utterances_from_csv();
    //std::cout << "finished collecting csv utterances " << utterances2.size() << std::endl;
    //utterances.insert(utterances.end(), utterances2.begin(), utterances2.end());
    std::vector<std::string> utterances3 = parse_utterances_from_dialog_csv();
    std::cout << "finished collecting dialog csv utterances " << utterances3.size() << std::endl;
    utterances.insert(utterances.end(), utterances3.begin(), utterances3.end());
    std::cout << "total utterances count " << utterances.size() << std::endl;

    std::unordered_map<std::string, std::unordered_map<std::string, int>> vocab_dist;

    for (const std::string &utterance : utterances) {    
        try {
            std::vector<std::string> tokens = tokenize(utterance);
            // add start tokens in utterance preprocessing
            for (int i = 0; i < n; i++) tokens.insert(tokens.begin(), "<start>");

            // loop through the tokens in the utterance and count up how many times each word comes after each ngram
            for (size_t i = n; i < tokens.size(); i++) {
                std::string ngram = tokens[i-n];
                for (int j = 1; j < n; j++) ngram += ("|" + tokens[i-n+j]);
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

    // filter out all ngrams seen less than some number of times in the training set
    filter_vocab(5, vocab_dist);

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    std::cout << "training time: " << cpu_time_used << "s" << std::endl;
    
    return vocab_dist;
}

std::string predict_next_word(std::unordered_map<std::string, std::unordered_map<std::string, int>> vocab_dist, std::string prev_word) {
    std::cout << "history " << prev_word << std::endl;
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
        std::cout << "vocab miss" << std::endl;
        // if the prior isnt in our vocab - pick a random word from the vocab
        auto it = vocab_dist.begin();
        std::advance(it, rand()%vocab_dist.size());
        return (it->first).substr(0, it->first.find("|"));
    }
}

std::string predict_next_sentence(std::unordered_map<std::string, std::unordered_map<std::string, int>> vocab_dist, std::vector<std::string> history, int n) {

    std::string output = "";
    std::string prediction;
    do {
        std::string ngram = "";
        for (int i = n; i > 0; i--) {
            ngram += history.at(history.size()-i);
            if (i > 1) ngram += "|";
        }
        prediction = predict_next_word(vocab_dist, ngram);
        output += prediction + " ";
        history.push_back(prediction);
    } while (prediction != "<stop>");

    return output;
}