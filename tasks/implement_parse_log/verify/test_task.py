from log_parser import parse_log_line


def test_valid_info():
    result = parse_log_line("2024-01-15T10:30:00 INFO Server started")
    assert result == {
        "timestamp": "2024-01-15T10:30:00",
        "level": "INFO",
        "message": "Server started",
    }


def test_valid_error():
    result = parse_log_line("2024-06-01T00:00:00 ERROR disk full")
    assert result["level"] == "ERROR"
    assert result["message"] == "disk full"


def test_invalid():
    assert parse_log_line("not a log line") is None
    assert parse_log_line("") is None
