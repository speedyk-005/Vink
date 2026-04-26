try:
    from uuid import uuid7
except ImportError:
    from uuid6 import uuid7  # Python <3.14


def generate_id() -> str:
    """Generate a UUIDv7 as a string.

    Returns:
        str: RFC 9562 UUIDv7 in standard string format.
    """
    return str(uuid7())


def generate_id_bytes() -> bytes:
    """Generate a UUIDv7 as 16 bytes.

    Returns:
        bytes: UUIDv7 in 16-byte binary form.
    """
    return uuid7().bytes
