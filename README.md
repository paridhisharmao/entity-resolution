# Entity Resolution

This project focuses on Entity Resolution for business records from multiple data sources.

## Dataset

The project contains:

- 3 training source files
- 3 test source files

Each record contains fields such as:

- entity_id
- business_name
- business_address
- country

## Current Approach

The current pipeline focuses on high-recall candidate generation using:

1. Chunked TSV reading
2. Text normalization
3. Address component extraction
4. Multi-pass blocking
5. SQLite-based indexing
6. Candidate generation
7. Training-set recall diagnostics

## Blocking

Multiple blocking keys are used, including:

- Country + business name
- Country + name tokens
- Postcode + name
- House number + name
- Street + name
- City + name
- State + name
- Phonetic and character-based keys

The goal of blocking is to reduce the number of unnecessary comparisons while retaining true matching records.

## Project Structure

```text
entity-resolution/
├── train/
├── test/
├── output/
├── main.py
├── README.md
└── requirements.txt