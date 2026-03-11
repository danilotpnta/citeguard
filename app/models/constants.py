from enum import Enum


class ContentType(str, Enum):
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    TEXT = "text/plain"

    @classmethod
    def from_mime(cls, mime: str) -> "ContentType":
        """Convert a raw MIME string to a ContentType. Raises ValueError if unsupported."""
        for member in cls:
            if member.value == mime:
                return member
        supported = ", ".join(m.name for m in cls)
        raise ValueError(f"Unsupported file type: {mime}. Accepted: {supported}")
    
    @property
    def short(self) -> str:
        return self.name.lower()  # PDF -> "pdf", DOCX -> "docx", TEXT -> "text"


class UploadConfig:
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    ALLOWED_TYPES = {
        ContentType.PDF,
        ContentType.DOCX,
        ContentType.TEXT,
    }
