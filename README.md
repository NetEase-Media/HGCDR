# HGCDR

# 1. Background
This project is the implementation for 'Heterogeneous Graph-based Framework with Disentangled Representations Learning for Multi-target Cross Domain Recommendation' 

Link to our paper: https://arxiv.org/abs/2407.00909

# 2. Install & Run
1. Make sure that your virtual env has the following packages:
   1. PyTorch (GPU Ver recommended)
      ```shell
      pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
      ```
   2. Numpy
   3. Pandas
   4. DGL (0.7.2 ver)
      ```shell
      pip install dgl-cu113==0.7.2 -f https://data.dgl.ai/wheels/repo.html
      ```
   5. Tensorboard
   6. Scikit-Learn


2. download the project from the repository
3. download the Douban data from the url: https://github.com/fengzhu1/GA-DTCDR/tree/main/Data, and move the dataset to the dir: ./data/doudan
4. run the command in the virtual env:
   if you want to run the Douban data
   ```shell
   python src/model_douban.py
   ```
   
5. the whole project support NVIDIA GPU ACCELERATION.
