conda create -n natspeech -c conda-forge montreal-forced-aligner -y
conda activate natspeech
pip install Cython numpy -i https://mirrors.aliyun.com/pypi/simple
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
