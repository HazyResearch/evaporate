from setuptools import find_packages, setup

_REQUIRED = [
    "tqdm",
    "openai",
    "manifest-ml",
    'beautifulsoup4',
    'pandas',
    'cvxpy',
    'sklearn',
    'scikit-learn',
    'snorkel',
    'snorkel-metal', 
    'tensorboardX',
    'pyyaml',
    'TexSoup',
]

setup(
    name="evaporate",
    version="0.0.1",
    description="evaporating data lakes with foundation models (Brando's version)",
    author="Brando",
    # https://chat.openai.com/share/1a431e8d-fa50-4663-b199-10f27bffa623
    # note evapoarte is its own package (a module), so it has a __init__.py and it's available as an import in python as import evaporate https://chat.openai.com/g/g-KV0CvoH8Y-python-excellent-comments-doc-strings-types/c/125c002a-3e77-41a9-8130-b3ae8da26d4c
    package_dir={'': 'evaporate'},  # Define the root directory for packages as 'evaporate'
    packages=find_packages('evaporate'),  # Automatically discover all packages in 'evaporate' directory
    install_requires=_REQUIRED,
)
