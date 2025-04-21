import pytest

# Filter out pydantic deprecation warnings from dependencies
# and urllib3 InsecureRequestWarning
def pytest_configure(config):
    """Configure pytest to ignore specific warnings."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore::DeprecationWarning:pydantic.*:"
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore::urllib3.exceptions.InsecureRequestWarning"
    ) 