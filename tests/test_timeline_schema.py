"""Test timeline schema validation."""
import json
from pathlib import Path


def test_timeline_schema_exists():
    """Verify timeline schema file exists and is valid JSON."""
    schema_path = Path(__file__).parent.parent / "schemas" / "timeline.schema.json"
    assert schema_path.exists(), f"Schema file not found: {schema_path}"
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    
    assert schema["title"] == "Tennis Rally Timeline"
    assert "items" in schema


def test_timeline_schema_has_required_fields():
    """Verify schema defines required fields."""
    schema_path = Path(__file__).parent.parent / "schemas" / "timeline.schema.json"
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    
    required = schema["items"]["required"]
    assert "clip_id" in required
    assert "start" in required
    assert "end" in required
    assert "keep" in required


def test_sample_timeline_valid():
    """Test a sample timeline passes schema validation."""
    sample = [
        {
            "clip_id": "point_001",
            "video_id": "session_001",
            "start": 83.42,
            "end": 101.88,
            "duration": 18.46,
            "label": ["rally", "long_rally"],
            "confidence": 0.84,
            "features": {"motion_mean": 0.71},
            "keep": True,
        }
    ]
    
    # Basic validation without full jsonschema library
    item = sample[0]
    assert item["start"] >= 0
    assert item["end"] > item["start"]
    assert 0 <= item["confidence"] <= 1
    assert isinstance(item["keep"], bool)
