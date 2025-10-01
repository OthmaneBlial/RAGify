from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.handlers import bcrypt as passlib_bcrypt

from .config import settings


def _patch_passlib_bcrypt_wrap_detection() -> None:
    """Work around passlib+bcrypt incompatibility seen on CI runners."""

    patch_flag = "_ragify_wrap_detection_patched"
    if getattr(passlib_bcrypt, patch_flag, False):
        return

    original_detect = getattr(passlib_bcrypt, "detect_wrap_bug", None)
    if original_detect is None:
        setattr(passlib_bcrypt, patch_flag, True)
        return

    def detect_wrap_bug_safe(ident: bytes) -> bool:  # type: ignore[override]
        try:
            return original_detect(ident)
        except ValueError as exc:
            message = str(exc).lower()
            if "password cannot be longer than 72 bytes" in message:
                # Newer versions of the bcrypt backend raise instead of truncating
                # long passwords during passlib's safety probe. Treat this as
                # "no wrap bug" so hashing can proceed normally.
                return False
            raise

    passlib_bcrypt.detect_wrap_bug = detect_wrap_bug_safe  # type: ignore[assignment]
    setattr(passlib_bcrypt, patch_flag, True)


_patch_passlib_bcrypt_wrap_detection()

if settings.secret_key == "your-secret-key-here":
    raise RuntimeError("SECRET_KEY must be set")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Truncate password to 72 bytes as required by bcrypt
    password_bytes = plain_password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
        plain_password = password_bytes.decode('utf-8', errors='ignore')
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    # Truncate password to 72 bytes as required by bcrypt
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
        password = password_bytes.decode('utf-8', errors='ignore')
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key, algorithm=settings.algorithm
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        return payload
    except JWTError:
        return None
