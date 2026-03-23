from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pharmkg-dti",
    version="1.0.0",
    author="PharmKG-DTI Team",
    author_email="",
    description="Production-Ready Knowledge Graph System for Drug-Target Interaction Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ei1eenp0nda-claw/pharmkg-dti",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "pytest-cov>=3.0",
        ],
        "optuna": ["optuna>=3.0"],
        "viz": ["matplotlib>=3.5", "seaborn>=0.11"],
        "api": ["fastapi>=0.95", "uvicorn>=0.20"],
    },
    entry_points={
        "console_scripts": [
            "pharmkg-dti=src.cli:main",
        ],
    },
)
