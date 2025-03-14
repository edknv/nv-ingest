# flake8: noqa
from dotenv import load_dotenv

# load_dotenv(".env")

import time

from nv_ingest.util.pipeline.pipeline_runners import start_pipeline_subprocess
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_client.message_clients.simple.simple_client import SimpleClient
from nv_ingest.util.pipeline.pipeline_runners import PipelineCreationSchema
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

# Start the pipeline subprocess for library mode

config = PipelineCreationSchema()
print(config, flush=True)

pipeline_process = start_pipeline_subprocess(config)

client = NvIngestClient(
    message_client_allocator=SimpleClient,
    message_client_port=7671,
    message_client_hostname="localhost",
)

# Note: gpu_cagra accelerated indexing is not yet available in milvus-lite
# Provide a filename for milvus_uri to use milvus-lite
milvus_uri = "milvus.db"
collection_name = "test"
sparse = False

# do content extraction from files
ingestor = (
    Ingestor(client=client)
    .files("data/multimodal_test.pdf")
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_images=True,
        paddle_output_format="markdown",
        extract_infographics=True,
        text_depth="page",
    )
    .embed()
    .caption()
    .vdb_upload(
       collection_name=collection_name,
       milvus_uri=milvus_uri,
       sparse=sparse,
       # for llama-3.2 embedder, use 1024 for e5-v5
       dense_dim=2048
    )
)

print("Starting ingestion..")
t0 = time.time()
results = ingestor.ingest()
t1 = time.time()
print(f"Time taken: {t1-t0} seconds")

# results blob is directly inspectable
print(ingest_json_results_to_blob(results[0]))
