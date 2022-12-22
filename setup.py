import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reqomp",
    version="0.0.1",
    description="Automatically synthesize uncomputation in a given quantum circuit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
	install_requires=[
		'qiskit==0.39.0', 
		'numpy==1.23.1', 
        'matplotlib==3.6.2'
	]
)
