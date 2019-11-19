# TAXONS

Task Agnostic eXploration of Outcome spaces through Novelty and Surprise.

This is the code of the paper: [Unsupervised Learning and Exploration of Reachable Outcome Space
](https://arxiv.org/abs/1909.05508)

---

To install run:
```
pipenv shell --three
python setup.py install
```

We also provide a containerized version running in Singularity at: https://github.com/GPaolo/taxons_sif

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
TAXONS needs Fastsim to be installed. I am using a slightly modified version of the original.

To install it we first need to download Pyfastsim:
```bash
git clone https://github.com/alexendy/pyfastsim
```

Now we need to install `libfastsim` in pyfastsim. To do so, execute:
```bash
cd external/pyfastsim
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
Now we can install pyfastsim by running, in the `external/pyfastsim` folder:
```
python setup.py install
```

**Finally we can install `fastsim-gym`**.
To do so, activate the virtual env and enter the `external` folder. Then do:
```.env
cd external
git clone https://github.com/GPaolo/fastsim_gym.git
git checkout patch-1
git pull origin patch-1
python setup.py install
```

---



The file `.env` will be loaded automatically with `pipenv shell` or `pipenv run your_command` and the environment variables will be available.


***NB***: within Pycharm you need the plugin Env File to load it automatically (access **Env File** tab from the Run/Debug configurations).
**You will have to run PyCharm from the shell itself from inside the activated virtualenv**

## Running
To run the algorithm you just need to launch:
```bash
python scripy/train.py
```

If you want to change the experiment parameters, go to: `script/parameters.py`

To plot the results, just run:
```bash
python scripts/plot.py
```
