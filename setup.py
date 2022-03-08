import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open('requirements.txt', 'r') as fh:
    requirements = fh.read().splitlines()

print(requirements)

setuptools.setup(
    name="gpz",
    version="1.0.0",
    author="Paul FELIOT",
    author_email="p.feliot@hotmail.fr",
    description="Gaussian process regression.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=['gpz'],
    install_requires = requirements,
    python_requires='>=3.6',
)
