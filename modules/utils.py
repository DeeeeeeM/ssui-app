def normalize_file_path(file_input):
    if not file_input:
        return None
    if isinstance(file_input, str):
        return file_input
    if isinstance(file_input, dict):
        for k in ("name", "tmp_path", "tempfile", "file_path", "path"):
            if k in file_input and file_input[k]:
                return file_input[k]
        return None
    try:
        return getattr(file_input, "name", None)
    except Exception:
        return None