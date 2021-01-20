




## Release Instructions

Build the distribution
```
python3 setup.py sdist bdist_wheel
```

Ship to PyPi
```
python3 -m twine upload --repository pypi  dist/*
```

To test a shipment to TestPyPI, use the following: 
```
python3 -m twine upload --repository testpypi dist/*
```
