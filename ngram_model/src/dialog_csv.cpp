#include "data_loaders/dialog_csv.h"

// Helper function to parse a CSV line into fields.
// It handles fields wrapped in double quotes (and doubled double quotes as escapes).
std::vector<std::string> parseCSVLine(const std::string &line) {
    std::vector<std::string> fields;
    std::string field;
    bool inQuotes = false;
    
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        
        if (inQuotes) {
            if (c == '"') {
                // Check for escaped quote
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    field.push_back('"');
                    ++i; // Skip the escaped quote
                } else {
                    inQuotes = false;
                }
            } else {
                field.push_back(c);
            }
        } else {
            if (c == '"') {
                inQuotes = true;
            } else if (c == ',') {
                fields.push_back(field);
                field.clear();
            } else {
                field.push_back(c);
            }
        }
    }
    fields.push_back(field);
    return fields;
}

// Function to parse the utterances from a line.
// Assumes that each line is in the form: <id>,<utterance1>,<utterance2>
std::pair<std::string, std::string> parseUtterances(const std::string &line) {
    std::vector<std::string> fields = parseCSVLine(line);
    
    // Check if there are at least three fields.
    if (fields.size() < 3) {
        std::cerr << "Error: Line does not contain enough fields: " << line << std::endl;
        return {"", ""};
    }
    
    // fields[0] is the id; fields[1] and fields[2] are the utterances.
    return {fields[1], fields[2]};
}