class FileTooLargeError(Exception):
    """Raised when an uploaded file exceeds the allowed size limit."""

    def __init__(self, limit_mb: int):
        self.limit_mb = limit_mb
        super().__init__(f"File exceeds the maximum allowed size of {limit_mb} MB.")


class InvalidFileTypeError(Exception):
    """Raised when an uploaded file's MIME type is not supported."""

    def __init__(self, valid_types: list):
        self.valid_types = valid_types
        super().__init__(
            f"Invalid file type. Supported types: {', '.join(valid_types)}."
        )


class ModelNotReadyError(Exception):
    """Raised when inference is attempted before the model has been loaded."""

    def __init__(self):
        super().__init__("The model is not loaded yet. Please try again later.")
