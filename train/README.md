# 训练

1. 转化训练集
将之前`/baker/render`文件夹中的文件，拷贝到`/train/dataset_raw`文件夹下，
然后在train目录下执行 `python loader.py` 转化训练集。执行完成后会在`/train/dataset/`下生成一个npz文件，并在`/train`目录下生成一下训练集的预览图

2. 训练
执行 `python train.py`，加载一个npz训练集开始训练，会在`/train/model`目录下生成checkpoint

3. 生成Lightmap
上一步训练完成后，
将之前`/baker/render`文件夹中生成的position.exr和normal.exr拷贝至`/train/lightmap`目录下，
然后执行`python test_lightmap.py` 生成lightmap，会在 `/train`目录下出现lightmap_0.0.webp和lightmap_1.0.webp两张图片

4. 生成shader
