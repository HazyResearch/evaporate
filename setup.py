from setuptools import setup

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
]

setup(
    name="evaporate",
    version="0.0.1",
    description="evaporating data lakes with foundation models",
    author="simran brandon sabri avanika andrew immanuel chris",
    packages=["evaporate"],
    install_requires=_REQUIRED,
)
