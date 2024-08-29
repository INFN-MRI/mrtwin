"""Utils for file download."""

__all__ = "ssl_verification"

import requests
from contextlib import contextmanager

@contextmanager
def ssl_verification(verify=True):  # noqa
    # Default behaviour (do not disable)
    if verify:
        yield

    # Store the original `requests.Session.send` method
    original_send = requests.Session.send

    # Define a new `send` method that disables SSL verification
    def send_with_ssl_disabled(self, *args, **kwargs):
        kwargs["verify"] = False
        return original_send(self, *args, **kwargs)

    # Replace the original `send` method with the new one
    requests.Session.send = send_with_ssl_disabled

    try:
        # Execute the code within the context
        yield
    finally:
        # Restore the original `send` method
        requests.Session.send = original_send