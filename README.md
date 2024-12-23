# Unsupervised Domain Adaptation on Point Cloud Classification via Imposing Structural Manifolds into Representation Space

### Instructions
Clone repo and install it
```bash
git clone https://github.com/Vencoders/PCUDA-MCC.git
cd PCUDA-MCC
pip install -r requirements.txt
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

Download data:
```bash
cd ./data
python download.py
```

Prepare data:
```bash
cd ./data
python download.py

python compute_norm_curv.py  # In line 149, note that the point number of pointcloud in shapenet is 1024, and in modelnet is 2048
python compute_norm_curv_scannet.py 
```


Training
```
python train_MCC.py 
Then
python train_DPFST.py
```
Testing
```
python test.py 
```

### Author
Li Yu {li.yu@nuist.edu.cn}
Hongchao Zhong {202212200013@nuist.edu.cn}
Longkun Zou {eelongkunzou@mail.scut.edu.cn}
Ke Chen {chenk02@pcl.ac.cn}



