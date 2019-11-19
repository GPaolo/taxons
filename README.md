# TAXONS

Task Agnostic eXploration of Outcome spaces through Novelty and Surprise.

This is the code of the paper: [Unsupervised Learning and Exploration of Reachable Outcome Space
](https://arxiv.org/abs/1909.05508)

---

To install run:
```
pipenv shell --three
pip install -e . --process-dependency-links
```
---
## Dependencies
**NB**: if you're using the virtualenv, activate it before installing the dependencies. 
### Pybullet Gym
I am using a slightly modified version of pybulletgym than the original found here: `https://github.com/benelot/pybullet-gym`.
 
To install it, activate the virtual env, go in the `external` folder and run:

```
git clone https://github.com/GPaolo/pybullet-gym.git
cd pybullet-gym
pip install -e .
```

If you want more informations, look at the `README` there.

### Fastsim
Also for this one I am using a slightly modified version of it. The original can be found here: `https://github.com/alexendy/fastsim_gym`.

Fastsim needs libfastsim to be installed first.

libfastsim needs to be install in pyfastsim, then patched with `patch -p1 < /path/to/your/file.patch`. Once this has been done you can install pyfastsim, then install fastsim-gym.

##### Pyfastsim
You can download it from `https://github.com/alexendy/pyfastsim`

##### Libfastsim
To install it, activate the virtual env and enter the `external/pyfastsim` folder. Then do:
```
git clone https://github.com/GPaolo/libfastsim.git
cd libfastsim
git checkout patch-1
git pull origin patch-1
patch -p1 < ../fastsim-boost2std-fixdisplay.patch
./waf configure
./waf build
./waf install
```
**NB** If it complains that cannot find boost, then install it by running:
```.env
sudo apt-get install libboost-all-dev
```

then go in the pyfastsim folder and install it by doing:
```
cd ..
python setup.py install
```

##### Fastsim-gym

To install it, activate the virtual env and enter the `external` folder. Then do:
```.env
git clone https://github.com/GPaolo/fastsim_gym.git
git checkout patch-1
git pull origin patch-1

```



---



The file `.env` will be loaded automatically with `pipenv shell` or `pipenv run your_command` and the environment variables will be available.


***NB***: within Pycharm you need the plugin Env File to load it automatically (access **Env File** tab from the Run/Debug configurations).
**You will have to run PyCharm from the shell itself from inside the activated virtualenv**

## Training
To run the algorithm you just need to launch:
```bash
python scripy/train.py
```

If you want to change the experiment parameters, go to: `script/parameters.py`

To plot the results, just run:
```bash
python scripts/plot.py
```