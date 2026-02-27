from functools import wraps
from pydantic import ConfigDict, ValidationError, validate_call

from vink.exceptions import InvalidInputError


def pretty_errors(error: ValidationError) -> str:
    """Formats Pydantic validation errors into a human-readable string."""
    lines = [
        f"{error.error_count()} validation error for {getattr(error, 'subtitle', '') or error.title}."
    ]
    for ind, err in enumerate(error.errors(), start=1):
        msg = err["msg"]

        loc = err.get("loc", [])
        formatted_loc = ""
        if len(loc) >= 1:
            formatted_loc = str(loc[0]) + "".join(f"[{step!r}]" for step in loc[1:])
            formatted_loc = f"({formatted_loc})" if formatted_loc else ""

        input_value = err["input"]
        input_type = type(input_value).__name__

        # Sliced to avoid overflowing screen
        input_value = (
            input_value
            if len(str(input_value)) < 500
            else str(input_value)[:500] + "..."
        )

        lines.append(
            (
                f"{ind}) {formatted_loc} {msg}.\n"
                f"  Found: (input={input_value!r}, type={input_type})"
            )
        )

    lines.append("  " + getattr(error, "hint", ""))
    return "\n".join(lines)


def validate_input(fn):
    """
    A decorator that validates function inputs and outputs

    A wrapper around Pydantic's `validate_call` that catches`ValidationError` and re-raises it as a more user-friendly `InvalidInputError`.
    """
    validated_fn = validate_call(fn, config=ConfigDict(arbitrary_types_allowed=True))  # type: ignore[call-overload]

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return validated_fn(*args, **kwargs)
        except ValidationError as e:
            raise InvalidInputError(pretty_errors(e)) from None

    return wrapper
