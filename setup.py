from setuptools import setup, find_packages

setup(
    name="litellm-kamiwaza",
    packages=find_packages(),
    install_requires=[
        "litellm>=1.0.0",
        "kamiwaza>=0.1.0", # provides kamiwaza_client
    ],
)
