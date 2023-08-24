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
    description="evaporating data lakes with foundation models",
    author="simran brandon sabri avanika andrew immanuel chris",
    # https://chat.openai.com/share/1a431e8d-fa50-4663-b199-10f27bffa623
    package_dir={'': 'evaporate'},  # Define the root directory for packages as 'evaporate'
    packages=find_packages('evaporate'),  # Automatically discover all packages in 'evaporate' directory
    install_requires=_REQUIRED,
)
