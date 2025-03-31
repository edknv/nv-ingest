import subprocess
import sys
import time

import pytest
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client import NvIngestClient


def test_images_extract_only(
    pipeline_process,
):
    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )

    ingestor = (
        Ingestor(client=client)
        .files("./data/multimodal_test.wav")
        .extract(
            extract_method="audio",
        )
    )

    results = ingestor.ingest()

    assert len(results) == 1

    print(results)
