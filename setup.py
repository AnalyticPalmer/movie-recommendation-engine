from pathlib import Path

from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent


setup(
    name="recommendation-engine",
    version="0.1.0",
    description="Production-oriented recommendation system project.",
    long_description=(ROOT_DIR / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.12",
)
