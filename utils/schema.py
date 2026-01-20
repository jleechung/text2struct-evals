"""JSON schema validation for per-structure outputs."""

from __future__ import annotations

from jsonschema import ValidationError, validate


OUTPUT_SCHEMA = {
    "type": "object",
    "required": [
        "structure_id",
        "structure_path",
        "accession",
        "sample_num",
        "eval_type",
        "status",
        "error",
        "predictions",
        "raw_output_path",
        "timestamp",
        "runtime_seconds",
    ],
    "properties": {
        "structure_id": {"type": "string"},
        "structure_path": {"type": "string"},
        "accession": {"type": "string"},
        "sample_num": {"type": "integer"},
        "eval_type": {
            "type": "string",
            "enum": ["dssp", "proteina", "p2rank", "clean", "thermompnn"],
        },
        "status": {"type": "string", "enum": ["success", "failed"]},
        "error": {"type": ["string", "null"]},
        "predictions": {"type": ["object", "null"]},
        "raw_output_path": {"type": ["string", "null"]},
        "timestamp": {"type": "string"},
        "runtime_seconds": {"type": "number"},
    },
    # Leave additionalProperties unspecified so tools can add fields later if needed.
}


def validate_output_json(output: dict) -> None:
    """Validate output JSON against OUTPUT_SCHEMA.

    Raises:
        ValueError: if invalid.
    """
    try:
        validate(instance=output, schema=OUTPUT_SCHEMA)
    except ValidationError as e:
        raise ValueError(f"Invalid output JSON: {e.message}") from e
