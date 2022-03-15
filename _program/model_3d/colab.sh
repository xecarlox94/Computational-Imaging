
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

!cp '/content/gdrive/My Drive/Colab Notebooks/data/archive.tar.gz' archive.tar.gz && \
  apt install vim && \
  tar -xzvf archive.tar.gz && \
  wget https://github.com/xecarlox94/Computational-Imaging/archive/refs/heads/master.zip && \
  unzip master && \
  mv Computational-Imaging-master Computational_Imaging && \
  mv dataset/ Computational_Imaging/model_3d/ && \
  rm *zip *gz

!sed -i 's/\.\/dataset/\/content\/Computational_Imaging\/model_3d\/dataset/g' /content/Computational_Imaging/model_3d/ml_train_dataset.py

