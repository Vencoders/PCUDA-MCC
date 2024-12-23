# Unsupervised Domain Adaptation on Point Cloud Classification via Imposing Structural Manifolds into Representation Space

### Instructions
Clone repo and install it
```bash
git clone https://github.com/Vencoders/PCUDA-MCC.git
cd gast
pip install -r requirements.txt
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



