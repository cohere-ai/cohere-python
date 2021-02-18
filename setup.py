import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cohere",
    version="0.0.5",
    author="kipply",
    author_email="carol@cohere.ai",
    description="A Python library for the Cohere API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cohere-ai/cohere-python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
