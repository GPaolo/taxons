# RND_QD

Quality-Diversity with Random Network Distillation as surprise metric

To install run:
```
pipenv shell --three
pip install -e . --process-dependency-links
```

---
If you get issues in installing `mujoco-py` in the virtualenv, do the following:

```bash
cd rnd_qd
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin >.env # create the file and write into
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-410 >>.env    # append to the file
echo LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-410/libGL.so >>.env
```
The file `.env` be loaded automatically with `pipenv shell` or `pipenv run your_command` and the environment variables will be available.


***NB***: within Pycharm you need the plugin Env File to load it automatically (access **Env File** tab from the Run/Debug configurations).
**You will have to run PyCharm from the shell itself from inside the activated virtualenv**

```bash
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3      
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
sudo apt-get install libglew-dev        
pip install mujoco-py
```

To test the installation interactively, launch python and follow these steps:
```
import mujoco_py
import os
mj_path, _ = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

sim.step()
print(sim.data.qpos)
# [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
#   7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
#   8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
#  -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
#  -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
#  -2.22862221e-05]
```