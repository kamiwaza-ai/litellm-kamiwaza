#!/bin/bash
# Script to build and publish the litellm-kamiwaza package

# Ensure script fails on error
set -e

echo "===== Building litellm-kamiwaza package ====="

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Install build tools if needed
echo "Installing/updating build tools..."
pip install --upgrade pip build twine

# Build the package
echo "Building the package..."
python -m build

# Show what we have
echo "Generated package files:"
ls -lh dist/

# Confirm before upload
read -p "Ready to publish to PyPI? (y/n) " ready_to_publish

if [ "$ready_to_publish" != "y" ]; then
    echo "Aborting PyPI publish."
    exit 0
fi

# Upload to PyPI
echo "Uploading to PyPI..."
python -m twine upload dist/*

echo "===== Package published successfully! ====="
echo "Users can now install with: pip install litellm-kamiwaza"
