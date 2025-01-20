# RAG Evaluation Library

A comprehensive library for evaluating Retrieval-Augmented Generation (RAG) models, with a focus on measuring citation accuracy, reasoning quality, and content reliability.

## Overview

This library provides automated evaluation metrics for RAG systems, currently optimized for the Pleias model format. It combines traditional text analysis techniques with LLM-based evaluation to provide an assessment of RAG performance.

## Features

The library calculates a RAG Index based on multiple evaluation metrics:

### Text-Based Metrics

| Metric | Description | Method |
|--------|-------------|---------|
| Non-hallucinated Citation | Reprinting ratio of the citation from the claimed source. Low reprinting ratio means the text has not me literally quoted and is likely hallucinated. | Smith-Waterman reprinting ratio => share of continuous strings that are identical in both texts, regardless of punctuation. |
| Valid Quote | Share of quotes that are actual texts and not identifiers or other non-valid texts. | Regular expression matching at least 3 words. |
| Valid Identifier | Share of quotes that have an identifier matching the one used for the sources being sent. | Source ID validation |
| Unduplicated Quote | Share of quotes that are not duplicated multiple times. | Regular expression matching. |

### LLM-as-Judge Metrics (using finetuned llama model, to be made available on Hugging Face)

| Metric | Description |
|--------|-------------|
| Query Adherence | Share of texts where the answer does fit the original query from the user either fully or partially. We further exclude from this count the case where the model refuses to answer (for instance due to irrelevant sources or inability to parsed them). |
| Grounded Statement | Share of statements associated to a quotation that are actually verified and grounded by the quotation. |
| Language Quality | | Valid Identifier | Share of quotes that have an identifier matching the one used for the sources being sent. | Source ID validation |
 |
| Reasoning Quality | Share of answers with a solid or generally correct reasoning structure and argumentative chaining. |

## RAG Index

The final RAG Index is calculated as the mean of all available evaluation metrics.

## Format Requirements

The library expects the input data to be in a parquet file format, with the following columns:

- `text`: The source context (including tagged sources)
- `generated_response`: The model's response

There are two formats for the input data. These are the special token format used by the pleias models, and then a more generic (but still structured) format that can be used with any model. 

The special token format is as follows:

<|query_start|> query text <|query_end|> <|source_start|> <|source_id_start|> source id <|source_id_end|> source text <|source_end|> <|analysis_start|> analysis text <|analysis_end|> <|answer_start|> answer text <|answer_end|>

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

The main difference between special_tokens and vllm is the format of the input data, as well as vllm using a chat format. The resulting dataframe can then be used for evaluation.

### Evaluation

The library provides comprehensive evaluation capabilities through two main components: hallucination detection and LLM-based evaluation.

### Requirements

- A parquet file containing the following columns:
  - `text`: The source context (including tagged sources)
  - `generated_response`: The model's response

### Basic Usage

```python
from rag_evaluation import RAGEvaluationPipeline

# Initialize evaluator
evaluator = RAGEvaluationPipeline(
    model_path="path/to/evaluation/model",
    max_model_len=8192  # optional
)

# Run evaluation
results = evaluator.evaluate(
    input_file="path/to/generations.parquet",
    output_file="path/to/save/results.parquet"  # optional
)

# Access results
print("\nEvaluation Results Summary:")
print("\nLLM-based Metrics:")
for key, value in results['llm_metrics'].items():
    print(f"{key}: {value:.3f}")

print("\nHallucination Metrics:")
for key, value in results['hallucination_metrics'].items():
    print(f"{key}: {value:.3f}")

print("\nOverall RAG Score:", f"{results['overall_rag_score']:.3f}")
```


