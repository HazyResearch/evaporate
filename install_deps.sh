# -- Create conda env for evaporate
conda create --name evaporate python=3.10
# conda create --name evaporate python=3.8
conda activate evaporate

# -- Evaporate code
cd ~
# cd $AFS
# echo $AFS
git clone git@github.com:brando90/evaporate.git
# ln -s /afs/cs.stanford.edu/u/brando9/evaporate $HOME/evaporate
cd ~/evaporate
pip install -e .

# -- Install missing dependencies not in setup.py
pip install tqdm openai manifest-ml beautifulsoup4 pandas cvxpy sklearn scikit-learn snorkel snorkel-metal tensorboardX

# -- Weak supervision code
cd ~/evaporate/metal-evap
git submodule init
git submodule update
pip install -e .

# -- Manifest (to install from source, which helps you modify the set of supported models. Otherwise, ``setup.py`` installs ``manifest-ml``)
cd ~cd ~
# cd $AFS
# echo $AFS
git clone git@github.com:HazyResearch/manifest.git
# ln -s /afs/cs.stanford.edu/u/brando9/manifest $HOME/manifest
cd ~/manifest
pip install -e .
pip install -e ~/manifest

# -- Git lfs install is a command used to initialize Git Large File Storage (LFS) on your machine.
git lfs install
# cd ~/evaporate/
cd ~/data
git clone https://huggingface.co/datasets/hazyresearch/evaporate
git lfs install
# get data in python
# from datasets import load_dataset
# dataset = load_dataset("hazyresearch/evaporate")






# -- Some installs are missing in setup.py
# pip install -e ~/evaporate/
pip install tqdm openai manifest-ml beautifulsoup4 pandas cvxpy sklearn scikit-learn snorkel snorkel-metal tensorboardX
pip install -e ~/evaporate/

pip install -e ~/manifest/cd ~/evaporate/metal-evap
git submodule init
git submodule update
pip install -e ~/evaporate/metal-evap
cd ~/evaporate 