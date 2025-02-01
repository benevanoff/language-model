#include "ngram_model.h"
#include "data_loaders/dialog_csv.h"

void test_prediction() {
    std::unordered_map<std::string, std::unordered_map<std::string, int>> vocab_dist = calc_word_dist(3);
    weights_to_file("weights.txt", vocab_dist);

    std::vector<std::string> history = tokenize("how are you doing ?");
    history.push_back("<stop>");
    std::string prediction = predict_next_sentence(vocab_dist, history, 3);
    std::cout << "prediction: " << prediction << std::endl;
}

void test_dialog_csv() {
    std::string line1 = "0,\"hi, how are you doing?\",i'm fine. how about yourself?";
    std::string line2 = "1,i'm fine. how about yourself?,i'm pretty good. thanks for asking.";
    std::string line3 = "2,i'm pretty good. thanks for asking.,no problem. so how have you been?";
    
    auto utterances1 = parseUtterances(line1);
    auto utterances2 = parseUtterances(line2);
    auto utterances3 = parseUtterances(line3);
    
    std::cout << "Line 1:\n  Utterance 1: " << utterances1.first
              << "\n  Utterance 2: " << utterances1.second << "\n\n";
              
    std::cout << "Line 2:\n  Utterance 1: " << utterances2.first
              << "\n  Utterance 2: " << utterances2.second << "\n\n";
              
    std::cout << "Line 3:\n  Utterance 1: " << utterances3.first
              << "\n  Utterance 2; " << utterances3.second << std::endl;
}

int main() {
    
    test_prediction();

    return 0;
}