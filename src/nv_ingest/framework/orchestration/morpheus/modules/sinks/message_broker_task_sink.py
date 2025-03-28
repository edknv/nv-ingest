# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import logging
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import mrc
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops

from nv_ingest.framework.schemas.framework_message_broker_sink_schema import MessageBrokerTaskSinkSchema
from nv_ingest_api.internal.primitives.tracing.logging import annotate_cm
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest.framework.orchestration.morpheus.util.modules.config_validator import (
    fetch_and_validate_module_config,
)
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.util.service_clients.client_base import MessageBrokerClientBase
from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient

logger = logging.getLogger(__name__)

MODULE_NAME = "message_broker_task_sink"
MODULE_NAMESPACE = "nv_ingest"

MessageBrokerTaskSinkLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


def extract_data_frame(message: IngestControlMessage) -> Tuple[Any, Dict[str, Any]]:
    """
    Extracts a DataFrame from a message payload and returns it along with a filtered dictionary of required columns.

    Parameters
    ----------
    message : IngestControlMessage
        The message object containing the payload.

    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        A tuple containing the DataFrame and a dictionary of selected columns.
    """
    try:
        df = message.payload()
        logger.debug(f"Message broker sink Received DataFrame with {len(df)} rows.")
        keep_cols = ["document_type", "metadata"]

        return df, df[keep_cols].to_dict(orient="records")
    except Exception as err:
        logger.warning(f"Failed to extract DataFrame from message payload: {err}")
        return None, None


def split_large_dict(json_data: List[Dict[str, Any]], size_limit: int) -> List[List[Dict[str, Any]]]:
    """
    Splits a large list of dictionaries into smaller fragments, each less than the specified size limit (in bytes).

    Parameters
    ----------
    json_data : List[Dict[str, Any]]
        The list of dictionaries to split.
    size_limit : int
        The maximum size in bytes for each fragment.

    Returns
    -------
    List[List[Dict[str, Any]]]
        A list of fragments, each fragment being a list of dictionaries, within the size limit.
    """

    fragments = []
    current_fragment = []
    current_size = sys.getsizeof(json.dumps(current_fragment))

    for item in json_data:
        item_size = sys.getsizeof(json.dumps(item))

        # If adding this item exceeds the size limit, start a new fragment
        if current_size + item_size > size_limit:
            fragments.append(current_fragment)  # Store the current fragment
            current_fragment = []  # Start a new fragment
            current_size = sys.getsizeof(json.dumps(current_fragment))

        # Add the item (dict) to the current fragment
        current_fragment.append(item)
        current_size += item_size

    # Append the last fragment if it has data
    if current_fragment:
        fragments.append(current_fragment)

    return fragments


def create_json_payload(message: IngestControlMessage, df_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Creates JSON payloads based on message status and data. If the size of df_json exceeds 256 MB, splits it into
    multiple fragments, each less than 256 MB. Adds optional trace and annotation data to the first fragment.
    """
    # Convert df_json to a JSON string to check its size
    df_json_str = json.dumps(df_json)
    df_json_size = sys.getsizeof(df_json_str)

    # 256 MB size limit (in bytes)
    size_limit = 128 * 1024 * 1024

    # If df_json is larger than the size limit, split it into chunks
    if df_json_size > size_limit:
        data_fragments = split_large_dict(df_json, size_limit)
        fragment_count = len(data_fragments)
    else:
        data_fragments = [df_json]
        fragment_count = 1

    ret_val_json_list = []

    # Process each fragment and add necessary metadata
    for i, fragment_data in enumerate(data_fragments):
        ret_val_json = {
            "status": "success" if not message.get_metadata("cm_failed", False) else "failed",
            "description": (
                "Successfully processed the message."
                if not message.get_metadata("cm_failed", False)
                else "Failed to process the message."
            ),
            "data": fragment_data,
            "fragment": i,
            "fragment_count": fragment_count,
        }

        # Only add trace tagging and annotations to the first fragment
        if i == 0 and message.get_metadata("add_trace_tagging", True):
            # Use the snapshot of trace timestamps directly
            trace_snapshot = message.filter_timestamp("trace::")
            ret_val_json["trace"] = {key: ts.timestamp() * 1e9 for key, ts in trace_snapshot.items()}
            ret_val_json["annotations"] = {
                key: message.get_metadata(key) for key in message.list_metadata() if key.startswith("annotation::")
            }

        ret_val_json_list.append(ret_val_json)

    logger.debug(f"Message broker sink created {len(ret_val_json_list)} JSON payloads.")
    return ret_val_json_list


def push_to_broker(
    broker_client: MessageBrokerClientBase, response_channel: str, json_payloads: List[str], retry_count: int = 2
) -> None:
    """
    Attempts to push a JSON payload to a message broker channel, retrying on failure up to a specified number of
    attempts.

    Parameters
    ----------
    broker_client : MessageBrokerClient
        The broker client used to push the data.
    response_channel : str
        The broker channel to which the data is pushed.
    json_payload : str
        The JSON string payload to be pushed.
    retry_count : int, optional
        The number of attempts to retry on failure (default is 2).

    Returns
    -------
    None

    Raises
    ------
    Valuerror
        If pushing to the message broker fails after the specified number of retries.
    """

    for json_payload in json_payloads:
        payload_size = sys.getsizeof(json_payload)
        size_limit = 2**28  # 256 MB

        if payload_size > size_limit:
            raise ValueError(f"Payload size {payload_size} bytes exceeds limit of {size_limit / 1e6} MB.")

    for attempt in range(retry_count):
        try:
            for json_payload in json_payloads:
                broker_client.submit_message(response_channel, json_payload)

            logger.debug(f"Message broker sink forwarded message to broker channel '{response_channel}'.")

            return
        except ValueError as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retry_count - 1:
                raise


def handle_failure(
    broker_client: MessageBrokerClientBase,
    response_channel: str,
    json_result_fragments: List[Dict[str, Any]],
    e: Exception,
    mdf_size: int,
) -> None:
    """
    Handles failure scenarios by logging the error and pushing a failure message to a broker channel.

    Parameters
    ----------
    broker_client : Any
        A MessageBrokerClientBase instance.
    response_channel : str
        The broker channel to which the failure message will be sent.
    json_result_fragments : List[Dict[str, Any]]
        A list of JSON result fragments, where each fragment is a dictionary containing the results of the operation.
        The first fragment is used to extract trace data in the failure message.
    e : Exception
        The exception object that triggered the failure.
    mdf_size : int
        The number of rows in the message data frame (mdf) being processed.

    Returns
    -------
    None
        This function does not return any value. It handles the failure by logging the error and sending a message to
        the message broker.

    Notes
    -----
    The failure message includes the error description, the size of the first JSON result fragment in MB,
    and the number of rows in the data being processed. If trace information is available in the first
    fragment of `json_result_fragments`, it is included in the failure message.

    Examples
    --------
    >>> broker_client = RedisClient()
    >>> response_channel = "response_channel_name"
    >>> json_result_fragments = [{"trace": {"event_1": 123456789}}]
    >>> e = Exception("Network failure")
    >>> mdf_size = 1000
    >>> handle_failure(broker_client, response_channel, json_result_fragments, e, mdf_size)
    """
    error_description = (
        f"Failed to forward message to message broker after retries: {e}. "
        f"Payload size: {sys.getsizeof(json.dumps(json_result_fragments)) / 1e6} MB, Rows: {mdf_size}"
    )
    logger.error(error_description)

    # Construct a failure message and push it to the message broker
    fail_msg = {
        "data": None,
        "status": "failed",
        "description": error_description,
        "trace": json_result_fragments[0].get("trace", {}),
    }
    broker_client.submit_message(response_channel, json.dumps(fail_msg))


def process_and_forward(message: IngestControlMessage, broker_client: MessageBrokerClientBase) -> IngestControlMessage:
    """
    Processes a message by extracting data, creating a JSON payload, and attempting to push it to the message broker.

    Parameters
    ----------
    message : IngestControlMessage
        The message to process.
    broker_client : MessageBrokerClientBase
        The message broker client used for pushing data.

    Returns
    -------
    IngestControlMessage
        The processed message.

    Raises
    ------
    Exception
        If a critical error occurs during processing.
    """
    mdf = None
    json_result_fragments = []
    response_channel = message.get_metadata("response_channel")

    try:
        cm_failed = message.get_metadata("cm_failed", False)
        if not cm_failed:
            mdf, df_json = extract_data_frame(message)
            json_result_fragments = create_json_payload(message, df_json)
        else:
            json_result_fragments = create_json_payload(message, None)

        # Convert each fragment to a JSON string and log its size in MB while accumulating the total size.
        total_payload_size = 0
        json_payloads = []
        for i, fragment in enumerate(json_result_fragments, start=1):
            payload = json.dumps(fragment)
            size_bytes = len(payload.encode("utf-8"))
            total_payload_size += size_bytes
            size_mb = size_bytes / (1024 * 1024)
            logger.debug(f"Fragment {i} size: {size_mb:.2f} MB")
            json_payloads.append(payload)

        total_size_mb = total_payload_size / (1024 * 1024)
        logger.debug(f"Total JSON payload size: {total_size_mb:.2f} MB")

        annotate_cm(message, message="Pushed")
        push_to_broker(broker_client, response_channel, json_payloads)
    except ValueError as e:
        mdf_size = len(mdf) if mdf is not None and not mdf.empty else 0
        handle_failure(broker_client, response_channel, json_result_fragments, e, mdf_size)
    except Exception as e:
        err_msg = f"Critical error processing message: {e}"
        logger.exception(err_msg)

        mdf_size = len(mdf) if not mdf.empty else 0
        handle_failure(broker_client, response_channel, json_result_fragments, e, mdf_size)

    return message


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _message_broker_task_sink(builder: mrc.Builder) -> None:
    """
    Configures and registers a processing node for message handling, including message broker task sinking.

    Parameters
    ----------
    builder : mrc.Builder
        The modular processing chain builder to which the message broker task sink node will be added.

    Returns
    -------
    None

    Notes
    -----
    This setup applies necessary decorators for failure handling and trace tagging. The node is then registered as
    both an input and an output module in the builder, completing the setup for message processing and
    forwarding to the message broker. It ensures that all messages passed through this node are processed and forwarded
    efficiently with robust error handling and connection management to the message broker.
    """

    validated_config = fetch_and_validate_module_config(builder, MessageBrokerTaskSinkSchema)
    # Determine the client type and create the appropriate client
    client_type = validated_config.broker_client.client_type.lower()
    broker_params = validated_config.broker_client.broker_params

    if client_type == "redis":
        client = RedisClient(
            host=validated_config.broker_client.host,
            port=validated_config.broker_client.port,
            db=broker_params.get("db", 0),
            max_retries=validated_config.broker_client.max_retries,
            max_backoff=validated_config.broker_client.max_backoff,
            connection_timeout=validated_config.broker_client.connection_timeout,
            use_ssl=broker_params.get("use_ssl", False),
        )
    elif client_type == "simple":
        client = SimpleClient(
            host=validated_config.broker_client.host,
            port=validated_config.broker_client.port,
            max_retries=validated_config.broker_client.max_retries,
            max_backoff=validated_config.broker_client.max_backoff,
            connection_timeout=validated_config.broker_client.connection_timeout,
        )
    else:
        raise ValueError(f"Unsupported client_type: {client_type}")

    @traceable(MODULE_NAME)
    def _process_and_forward(message: IngestControlMessage) -> IngestControlMessage:
        """
        Wraps the processing and forwarding functionality with traceability and error handling.

        Parameters
        ----------
        message : IngestControlMessage
            The message to be processed and forwarded to the message broker.

        Returns
        -------
        IngestControlMessage
            The processed message, after attempting to forward to the message broker.
        """
        return process_and_forward(message, client)

    process_node = builder.make_node("process_and_forward", ops.map(_process_and_forward))
    process_node.launch_options.engines_per_pe = validated_config.progress_engines

    # Register the final output of the module
    builder.register_module_input("input", process_node)
    builder.register_module_output("output", process_node)
