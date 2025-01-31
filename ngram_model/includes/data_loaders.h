#pragma once

// Apache Arrow headers
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/status.h>
#include <arrow/table.h>

// Apache Parquet headers
#include <parquet/arrow/reader.h>
#include <parquet/exception.h>

std::vector<std::string> parse_utterances_from_csv() {

    std::vector<std::string> utterances;
    
    std::ifstream ifs;
    ifs.open("../data/combined.csv");

    std::string line;
    std::getline(ifs, line); // discard head

    while (ifs.good()) {    
        try {
            // read each line from the data file
            // each line should contain a single utterance from a single interlocuter
            std::getline(ifs, line);
            utterances.push_back(line.substr(line.find(":")+2));
        } catch (std::exception e) {}
    }

    return utterances;
}

// Function to read and iterate through the "dialogue" column
std::vector<std::string> iterate_dialogue(const std::shared_ptr<arrow::Table>& table) {
    std::vector<std::string> utterances;

    // Retrieve the "dialogue" column
    std::shared_ptr<arrow::ChunkedArray> dialogue_column = table->GetColumnByName("dialogue");
    if (dialogue_column == nullptr) {
        std::cerr << "Column 'dialogue' not found in the table." << std::endl;
        return utterances;
    }

    // Iterate over each chunk in the column (usually one chunk, but handling multiple)
    for (const auto& chunk : dialogue_column->chunks()) {
        // Cast the chunk to a ListArray
        auto list_array = std::static_pointer_cast<arrow::ListArray>(chunk);

        // Get the underlying StringArray from the ListArray
        auto string_array = std::static_pointer_cast<arrow::StringArray>(list_array->values());

        // Iterate through each row in the chunk
        for (int64_t i = 0; i < list_array->length(); ++i) {
            std::cout << "loaded another chunk (" << utterances.size() << "/" << table->num_rows() << ")" << std::endl;

            if (list_array->IsNull(i)) {
                std::cout << "Row " << i << ": NULL" << std::endl;
                continue;
            }

            // Get the start and end offsets for the list in this row
            int64_t start = list_array->value_offset(i);
            int64_t end = list_array->value_offset(i + 1);
            int64_t list_size = end - start;

            //std::cout << "Row " << i << " Dialogue (" << list_size << " turns): ";

            // Iterate through each element in the list
            for (int64_t j = start; j < end; ++j) {
                if (string_array->IsNull(j)) {
                    // std::cout << "[NULL] ";
                } else {
                    // std::cout << "\"" << string_array->GetString(j) << "\" ";
                    utterances.push_back(string_array->GetString(j));
                }
            }
            // std::cout << std::endl;
        }
    }
    return utterances;
}

// Function to read the Parquet file and return an Arrow Table
std::shared_ptr<arrow::Table> read_parquet_file(const std::string& filename) {
    // Initialize Arrow IO
    arrow::Result<std::shared_ptr<arrow::io::ReadableFile>> infile_res = 
        arrow::io::ReadableFile::Open(filename, arrow::default_memory_pool());

    if (!infile_res.ok()) {
        std::cerr << "Error opening file: " << infile_res.status().ToString() << std::endl;
        return nullptr;
    }
    std::shared_ptr<arrow::io::ReadableFile> infile = *infile_res;

    // Initialize Parquet reader
    arrow::Result<std::unique_ptr<parquet::arrow::FileReader>> parquet_reader_res = 
        parquet::arrow::OpenFile(infile, arrow::default_memory_pool());

    if (!parquet_reader_res.ok()) {
        std::cerr << "Error creating Parquet reader: " << parquet_reader_res.status().ToString() << std::endl;
        return nullptr;
    }
    std::unique_ptr<parquet::arrow::FileReader> parquet_reader = std::move(*parquet_reader_res);

    // Read the entire Parquet file into an Arrow Table
    std::shared_ptr<arrow::Table> table;
    arrow::Status status = parquet_reader->ReadTable(&table);
    if (!status.ok()) {
        std::cerr << "Error reading table: " << status.ToString() << std::endl;
        return nullptr;
    }

    std::cout << "Successfully read Parquet file: " << filename << std::endl;
    std::cout << "Total Rows: " << table->num_rows() << std::endl;
    return table;
}

std::vector<std::string> parse_utterances_from_soda_parquet() {
    // Path to your Parquet file
    std::string parquet_file = "../data/test.parquet";

    // Read the Parquet file into an Arrow Table
    std::shared_ptr<arrow::Table> table = read_parquet_file(parquet_file);
    if (table == nullptr) {
        std::cerr << "Failed to read the Parquet file." << std::endl;
        std::vector<std::string> err_result;
        return err_result;
    }

    // Iterate through the "dialogue" column
    return iterate_dialogue(table);
}