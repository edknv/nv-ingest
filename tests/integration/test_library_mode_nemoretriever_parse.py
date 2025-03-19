import sys
import time

from nv_ingest_client.client import Ingestor
from nv_ingest_client.message_clients.simple.simple_client import SimpleClient

from nv_ingest.util.pipeline.pipeline_runners import PipelineCreationSchema
from nv_ingest.util.pipeline.pipeline_runners import start_pipeline_subprocess


def test_library_mode_extract_pdf():

    config = PipelineCreationSchema()

    pipeline_process = start_pipeline_subprocess(config, stderr=sys.stderr, stdout=sys.stdout)

    time.sleep(10)

    if pipeline_process.poll() is not None:
        raise

    ingestor = (
        Ingestor(
            message_client_allocator=SimpleClient,
            message_client_port=7671,
            message_client_hostname="localhost",
        )
        .files("data/multimodal_test.pdf")
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            paddle_output_format="markdown",
            extract_method="nemoretriever_parse",
            text_depth="page",
        )
        .embed()
        .caption()
        .vdb_upload(
            collection_name="test",
            milvus_uri="milvus.db",
            sparse=False,
            dense_dim=2048,
        )
    )

    results = ingestor.ingest()

    assert len(results) == 1
    texts = [x for x in results[0] if x["metadata"]["content_metadata"]["type"] == "text"]
    text_contents = [x["metadata"]["content"] for x in texts]
    assert any("A sample document with headings and placeholder text" in x for x in text_contents)
    # table in markdown
    tables = [x for x in results[0] if x["metadata"]["table_metadata"]["subtype"] == "table"]
    table_contents = [x["metadata"]["table_metadata"]["table_content"] for x in tables]
    assert any("| Dog | Chasing a squirrel | In the front yard |" in x for x in table_contents)
    # chart labels
    charts = [x for x in results[0] if x["metadata"]["table_metadata"]["subtype"] == "charts"]
    charts_contents = [x["metadata"]["table_metadata"]["table_content"] for x in charts]
    assert any("Tweeter - Midrange - Midwoofer - Subwoofer" in x for x in table_contents)
