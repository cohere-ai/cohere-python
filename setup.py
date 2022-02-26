import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(foo):
        return True

setuptools.setup(
    name='cohere',
    version='1.2.3',
    author='kipply',
    author_email='carol@cohere.ai',
    description='A Python library for the Cohere API',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/cohere-ai/cohere-python',
    packages=setuptools.find_packages(),
    install_requires=[
        'requests'
    ],
    package_data={'': ['./cohere/tokenizer/*']},
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    dist_class=BinaryDistribution
)
