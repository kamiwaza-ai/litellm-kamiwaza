from setuptools import setup, find_packages

setup(
    name="litellm-kamiwaza",
    packages=find_packages(),
    install_requires=[
        # <1.107.0: litellm 1.107.0 was malicious (2026-10-21); raise only after review.
        "litellm>=1.50.0,<1.107.0",
        "kamiwaza>=0.3.3", # This package provides the kamiwaza_client module
    ],
)
