import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


CAPTION_FIELDS = ["human_caption", "gemini_caption"]


@dataclass
class SampleRecord:
    key: str
    image_path: Optional[str]
    caption: Optional[str]
    caption_source: Optional[str]          # which field the caption came from
    raw: Dict[str, Any] = field(default_factory=dict)  # full original entry for any extra fields


def _resolve_caption(
    entry: Dict[str, Any],
    caption_field: Optional[str],
    skip_token: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Return (caption, caption_source).

    If caption_field is given, only that field is tried.
    Otherwise, CAPTION_FIELDS are tried in order and the first
    non-empty, non-rejected value is returned.
    """
    fields_to_try = [caption_field] if caption_field else CAPTION_FIELDS

    for field_name in fields_to_try:
        raw_val = (entry.get(field_name) or "").strip()
        if not raw_val:
            continue
        if skip_token and raw_val.lower() == skip_token.lower():
            continue
        return raw_val, field_name

    return None, None


def load_records(
    input_json: str,
    caption_field: Optional[str] = None,
    skip_token: Optional[str] = "reject",
    require_image_exists: bool = False,
) -> List[SampleRecord]:
    """
    Load and parse records from a human_labels-style JSON file.

    Args:
        input_json:           Path to the input JSON file.
        caption_field:        Which caption field to use ("human_caption" or "gemini_caption").
                              If None, tries all known caption fields in order and uses the
                              first valid one.
        skip_token:           Caption value to treat as invalid/rejected (case-insensitive).
                              Set to None to disable filtering.
        require_image_exists: If True, exclude records whose image path does not exist on disk.

    Returns:
        A list of SampleRecord objects, each with:
            - key            : the original JSON key
            - image_path     : value of "frame_path" in the entry (may be None)
            - caption        : resolved caption string (may be None if empty / rejected)
            - caption_source : name of the field the caption came from (e.g. "human_caption")
            - raw            : the full original dict, for access to any extra fields
    """
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    with open(input_json, "r") as f:
        data: Dict[str, Any] = json.load(f)

    records: List[SampleRecord] = []

    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue

        caption, caption_source = _resolve_caption(entry, caption_field, skip_token)
        image_path: Optional[str] = entry.get("frame_path")

        # Optionally skip records whose image file is missing
        if require_image_exists and (not image_path or not os.path.exists(image_path)):
            continue

        records.append(SampleRecord(
            key=key,
            image_path=image_path,
            caption=caption,
            caption_source=caption_source,
            raw=entry,
        ))

    return records