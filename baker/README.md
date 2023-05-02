# 准备训练集

当你有了一个Blender场景之后，就可以准备训练集训练神经网络了。
这一步你需要
- 生成一些相机的位置
- 从每个相机位置渲染一些图片


## 1. 生成相机
打开案例场景，在脚本中运行generate_camera.py，生成相机
- bpy.ops.mesh.primitive_cube_add(size=4.0), 尺寸控制相机的分布球面半径
- subd.levels = 2 球面的细分级别，这个数字越高，生成的相机越多。2时会生成98个相机
- cam.lens 相机焦距，控制渲染范围

## 2. 渲染最终渲染结果，位置法线信息和相机位置
执行render_cams.py, 注意选择分辨率，可以在/render/render中生成渲染最终效果的图片；在/render/pn中生成物体位置和法线信息的exr图片
这一步以后，/render文件夹下面的就是训练集了
- /render/render下面是渲染图，
- /render/pn下面是训练输入数据，exr的通道里面包含position和normal
- /render/cam_pos.json是相机参数
