from setuptools import setup, find_packages

setup(
    name="litellm-kamiwaza",
    packages=find_packages(),
    install_requires=[
        # litellm 1.82.7/1.82.8 (2026-03-24) and 1.107.0 (2026-10-21) were malicious;
        # raise the cap only after review.
        "litellm>=1.50.0,!=1.82.7,!=1.82.8,<1.107.0",
        "kamiwaza>=0.3.3", # This package provides the kamiwaza_client module
    ],
)
