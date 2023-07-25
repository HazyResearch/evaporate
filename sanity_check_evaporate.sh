# -- Git lfs install is a command used to initialize Git Large File Storage (LFS) on your machine.
git lfs install
# cd ~/evaporate/
cd ~/data
git clone https://huggingface.co/datasets/hazyresearch/evaporate
git lfs install

ls -l ~/data/evaporate/data/
ls -l ~/data/evaporate/data/fda_510ks/

# -- data
BASE_DATA_DIR=~/data/evaporate/data
ls -l $BASE_DATA_DIR

cd $BASE_DATA_DIR/fda_510ks/
ls -l
tar -xzvf docs.tar.gz
ls -l
ls -l data/evaporate/
ls -l $BASE_DATA_DIR/fda_510ks/data/evaporate/fda-ai-pmas/510k