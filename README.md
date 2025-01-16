# RAG Evaluation Library

A comprehensive library for evaluating Retrieval-Augmented Generation (RAG) models, with a focus on measuring citation accuracy, reasoning quality, and content reliability.

## Overview

This library provides automated evaluation metrics for RAG systems, currently optimized for the Pleias model format. It combines traditional text analysis techniques with LLM-based evaluation to provide an assessment of RAG performance.

## Features

The library calculates a RAG Index based on multiple evaluation metrics:

### Text-Based Metrics

| Metric | Description | Method |
|--------|-------------|---------|
| Non-hallucinated Citation | Measures the literal quote accuracy from source texts | Smith-Waterman algorithm for string alignment |
| Valid Quote | Verifies quotes contain actual content (≥3 words) | Regex pattern matching |
| Valid Identifier | Checks source reference accuracy | Source ID validation |
| Unduplicated Quote | Identifies redundant citations | Duplicate detection |

### LLM-as-Judge Metrics (using finetuned llama model, to be made available on Hugging Face)

| Metric | Description |
|--------|-------------|
| Query Adherence | Evaluates answer relevance to original query |
| Grounded Statement | Verifies citation support for claims |
| Language Quality | Assesses linguistic accuracy across multiple languages |
| Reasoning Quality | Evaluates logical structure and argumentation |

## RAG Index

The final RAG Index is calculated as the mean of all available evaluation metrics, providing a comprehensive score for RAG system performance.

## Format Requirements

TODO

## Usage

The library consists of two main components: generation and evaluation of RAG content.

### Generation

The generation component allows you to create RAG outputs using different types of generators. Currently supported generators are:
- VLLM Generator
- Special Tokens Generator

Basic usage:
```python
from rag_evaluation import RAGGenerator

# Initialize generator
generator = RAGGenerator(
    generator_type="special_tokens",  # or "vllm"
    input_path="path/to/input.parquet",
    model_path="path/to/model",
    num_rows=500  # optional, process all rows if not specified
)

# Generate responses
results = generator.generate()

# Optionally save outputs
generator.save_outputs(results, output_dir="./outputs")
```

Generated outputs will include:
- A parquet file containing all generations
- A readable text file with samples of the generations

The main difference between special_tokens and vllm is the format of the input data. The special tokens format is TODO

### Evaluation (Coming Soon)

The evaluation component will provide metrics for assessing the quality of RAG outputs.

## Installation

TODO

## Contributing

Contributions are welcome! Please see our contributing guidelines for more information.

## License

TODO