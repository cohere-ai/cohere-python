import setuptools
from setuptools.command.install import install

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None


class InstallPlatlib(install):
    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib

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
    cmdclass={'bdist_wheel': bdist_wheel, 'install': InstallPlatlib}
)
