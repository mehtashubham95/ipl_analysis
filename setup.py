from setuptools import setup, find_packages

setup(
    name="ipl-analysis",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="IPL Player Analysis and Team Optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "xgboost>=1.6.0",
        "seaborn>=0.11.0",
        "matplotlib>=3.5.0",
        "scipy>=1.9.0",
        "openpyxl>=3.0.0",
        "python-dotenv>=0.19.0",
    ],
)
