from setuptools import setup, find_packages

setup(
    name="fcmethods",
    version="0.1.0",
    description="Functional connectivity analysis methods for BIDS datasets",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "scipy>=1.6.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.9"],
    },
)
