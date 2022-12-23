import setuptools
from setuptools.command.install import install
from setuptools.dist import Distribution

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()


class InstallPlatlib(install):

    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


class BinaryDistribution(Distribution):

    def is_pure(self) -> bool:
        return False

    def has_ext_modules(foo) -> bool:
        return True


setuptools.setup(name='cohere',
                 version='3.1.3',
                 author='1vn',
                 author_email='ivan@cohere.ai',
                 description='A Python library for the Cohere API',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 url='https://github.com/cohere-ai/cohere-python',
                 packages=setuptools.find_packages(),
                 install_requires=['requests'],
                 include_package_data=True,
                 classifiers=[
                     'Programming Language :: Python :: 3',
                     'License :: OSI Approved :: MIT License',
                     'Operating System :: OS Independent',
                 ],
                 python_requires='>=3.6',
                 distclass=BinaryDistribution,
                 cmdclass={'install': InstallPlatlib})
