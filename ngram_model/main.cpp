#include "ngram_model.h"

int main() {
    int N = 3;
    srand(time(NULL));
    // calculate the word distribution from the training set - aka unsupervised learning
    std::unordered_map<std::string, std::unordered_map<std::string, int>> vocab_dist = calc_word_dist(N);
    weights_to_file("weights.txt", vocab_dist);

    while (true) {
        // get prompt from console
        std::cout << "Prompt: ";    
        std::string prompt;
        std::getline(std::cin, prompt);

        std::vector<std::string> history = tokenize(prompt);
        history.push_back("<stop>");

        std::string prediction = predict_next_sentence(vocab_dist, history, N);
        std::cout << "Prediction: " << prediction << std::endl;
    }

    return 0;
}