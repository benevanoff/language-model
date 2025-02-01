#pragma once

// Apache Arrow headers
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/status.h>
#include <arrow/table.h>

// Apache Parquet headers
#include <parquet/arrow/reader.h>
#include <parquet/exception.h>

std::vector<std::string> parse_utterances_from_csv();

std::vector<std::string> parse_utterances_from_dialog_csv();

std::vector<std::string> iterate_dialogue(const std::shared_ptr<arrow::Table>& table);

std::shared_ptr<arrow::Table> read_parquet_file(const std::string& filename);

std::vector<std::string> parse_utterances_from_soda_parquet();