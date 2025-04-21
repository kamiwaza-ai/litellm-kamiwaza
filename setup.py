from setuptools import setup, find_packages

setup(
    name="litellm-kamiwaza",
    packages=find_packages(),
    install_requires=[
        "litellm>=1.0.0",
        "kamiwaza>=0.3.3.0", # This package provides the kamiwaza_client module
    ],
)
