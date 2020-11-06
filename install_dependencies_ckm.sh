pip3 install tensorboard
pip3 install urllib3==1.25.4
pip3 install torch==1.4.0
python3 setup.py build
python3 setup.py install

cd ..
wget http://s3.amazonaws.com/proteindata/data_pytorch/pfam.tar.gz
tar xzf pfam.tar.gz
mkdir data
mv ./pfam ./data/pfam

pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex
