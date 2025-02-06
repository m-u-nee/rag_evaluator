from setuptools import setup, find_packages

setup(
    name="rag_evaluator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "vllm",
        "tqdm",
        "nltk",
        "rank_bm25",
        
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for evaluating RAG systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="rag, evaluation, nlp",
    python_requires=">=3.7",
)
