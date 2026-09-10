# parse_log_line spec

Input: a single log line string.

Format: `TIMESTAMP LEVEL message`

- TIMESTAMP: `YYYY-MM-DDTHH:MM:SS` (no timezone)
- LEVEL: one of INFO, WARN, ERROR
- message: remaining text after LEVEL and one space

Return `{"timestamp": str, "level": str, "message": str}` or `None` if invalid.
