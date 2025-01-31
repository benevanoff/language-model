#include "data_utils.h"
#include "data_loaders.h"

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

    std::vector<std::string> utterances = parse_utterances_from_soda_parquet();
    std::cout << "finished collecting soda utterances " << utterances.size() << std::endl;
    std::vector<std::string> utterances2 = parse_utterances_from_csv();
    std::cout << "finished collecting csv utterances " << utterances2.size() << std::endl;
    utterances.insert(utterances.end(), utterances2.begin(), utterances2.end());
    std::vector<std::string> utterances3 = parse_utterances_from_dialog_csv();
    std::cout << "finished collecting dialog csv utterances " << utterances3.size() << std::endl;
    utterances.insert(utterances.end(), utterances3.begin(), utterances3.end());
    std::cout << "total utterances count " << utterances.size() << std::endl;

    std::unordered_map<std::string, std::unordered_map<std::string, int>> vocab_dist;

    for (const std::string &utterance : utterances) {    
        try {
            std::vector<std::string> tokens = tokenize(utterance);
            // add start and stop tokens in utterance preprocessing
            for (int i = 0; i < n; i++) tokens.insert(tokens.begin(), "<start>");
            tokens.push_back("<stop>");

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

int main() {
    int N = 3;
    srand(time(NULL));
    // calculate the word distribution from the training set - aka unsupervised learning
    std::unordered_map<std::string, std::unordered_map<std::string, int>> vocab_dist = calc_word_dist(N);
    // weights_to_file("weights.txt", vocab_dist);
    
    // initialize sentence with start token and a user provided word
    std::vector<std::string> history;
    for (int i = 0; i < N-1; i++) history.push_back("<start>");

    // prompt the user for an initial word to kick start the sentence generation
    std::cout << "Starting word: ";
    std::string start;
    std::getline(std::cin, start);
    history.push_back(start);
    
    while (true) {
        // predict the next word and print it to the console
        std::string history_string = history.at(0);
        for (size_t i = 1; i < history.size(); i++) history_string += ("|" + history.at(i));
        std::string next_word = predict_next_word(vocab_dist, history_string);
        std::cout << "prediction: " << next_word;
        // wait for any input from console to continue
        std::string dummy;
        std::getline(std::cin, dummy);
        // advance the context window, adding the predicted word to the history (aka auto regressive generation)
        history.erase(history.begin());
        history.push_back(next_word);
    }
    return 0;
}