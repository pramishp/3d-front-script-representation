conda create -n lisa python=3.11

# check if conda init is required
if ! grep -q "conda initialize" ~/.bashrc && ! grep -q "conda" ~/.condarc; then
    echo "Conda is not initialized. Initializing now..."
    conda init bash < /dev/null
    source ~/.bashrc
else
    echo "Conda is already initialized."
fi
conda activate lisa

# installation
#pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
conda install nvidia/label/cuda-12.6.3::cuda-toolkit
pip install flash-attn --no-build-isolation

pip install -r requirements.txt


pip install deepspeed
pip install peft==0.4.0

conda install conda-forge::tensorboard
pip install numpy==1.24.2
pip install scipy==1.11.2
pip install scikit-image==0.21.0

pip install sentencepiece
#conda install -c conda-forge scikit-image

mkdir dataset
cd dataset && wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip -O ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip

# download SAM models
mkdir downloads
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O downloads/sam.pth
