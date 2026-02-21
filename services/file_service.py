import hashlib

from fastapi import UploadFile

from core.exceptions import FileTooLargeError, InvalidFileTypeError

_MAX_SIZE_MB = 5
_MAX_SIZE_BYTES = _MAX_SIZE_MB * 1024 * 1024
_VALID_MIME_TYPES = ["image/jpeg", "image/png"]


class FileService:
    @staticmethod
    async def validate_and_hash(file: UploadFile) -> str:
        """
        Validate an uploaded file's MIME type and size, then return its SHA-256 hash.

        Steps:
            1. Check MIME type — raises InvalidFileTypeError if not jpeg/png.
            2. Read content and check size — raises FileTooLargeError if over 5 MB.
            3. Compute SHA-256 hash.
            4. Seek pointer back to 0 so the file can be read again downstream.
            5. Return the hex digest.

        Args:
            file: The FastAPI UploadFile object from the incoming request.

        Returns:
            SHA-256 hex digest string of the file content.
        """
        # Step 1: MIME type check
        if file.content_type not in _VALID_MIME_TYPES:
            raise InvalidFileTypeError(valid_types=_VALID_MIME_TYPES)

        # Step 2: Read content and size check
        content = await file.read()
        if len(content) > _MAX_SIZE_BYTES:
            raise FileTooLargeError(limit_mb=_MAX_SIZE_MB)

        # Step 3: SHA-256 hash
        sha256_hash = hashlib.sha256(content).hexdigest()

        # Step 4: Reset file pointer so the model inference can read the file again
        await file.seek(0)

        # Step 5: Return hash
        return sha256_hash
