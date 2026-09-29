# Starter Kits for NeMo Retriever Library

Explore ready-made Jupyter notebooks and guides for [NeMo Retriever Library](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/). Examples cover document ingestion, custom vector database operators, and multimodal RAG with LangChain and LlamaIndex.

## Dataset Downloads for Benchmarking

If you run NeMo Retriever Library benchmarking or evaluation tests that use the Bo20, Bo767, or Bo10k corpora, download the [Benchmark Datasets](https://github.com/NVIDIA/NeMo-Retriever/blob/main/evaluation/digital_corpora_download.ipynb) from Digital Corpora first. The standalone ViDoRe worked example below downloads its own dataset.

## Getting Started

Start with these guides and notebooks:

- [Rerank Existing Retrieval Results (Text Only)](reranking/reranking_existing_retrieval_results_text_only.ipynb) — start here for a small-fixture example with Nemotron 3.5 Rerank 8B. It compares ranking quality without installing NeMo Retriever Library; first-run model download and optional embedding retrieval take longer. Requires a suitable local GPU.
- [Rerank ViDoRe v3 HR (Text Only)](reranking/reranking_vidore_v3_hr_text_only.ipynb) — compare BM25 with local, text-only Nemotron 3.5 Rerank 8B over all 1,110 pages and 1,908 queries in the HR dataset (not all ViDoRe v3 datasets). This can take a long time and requires a suitable local GPU. It does not use the Bo benchmarking datasets above.
- [Quickstart: retriever CLI](https://docs.nvidia.com/nemo/retriever/latest/reference/retriever-cli-quickstart/)
- [Workflow: Ingest documents](https://docs.nvidia.com/nemo/retriever/latest/extraction/workflow-document-ingestion/)
- [Adding Custom Metadata for Filtered Search/Retrieval](nemo_retriever_retriever_query_metadata_filter.ipynb) — also summarized on [Vector databases — Metadata and filtering](https://docs.nvidia.com/nemo/retriever/latest/extraction/vdbs/#metadata-and-filtering)

For advanced scenarios, use these guides and notebooks:

- [Build a Custom Vector Database Operator](building_vdb_operator.ipynb)
- [Try Enterprise RAG Blueprint](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag)
- [Multimodal RAG with LangChain](langchain_multimodal_rag.ipynb)
- [Multimodal RAG with LlamaIndex](llama_index_multimodal_rag.ipynb)
