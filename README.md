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

### LLM-as-Judge Metrics (using GPT-4)

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

