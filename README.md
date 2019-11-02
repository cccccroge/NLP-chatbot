# NLP_application
An easy NLP application running on Android system, mainly focus on helping students to write their life diary in a fun way.

# Project Development
## Installation
### Windows
[official guide from Kivy.org](https://kivy.org/doc/stable/installation/installation-windows.html)
1. Install Python 3.7
2. Setup pip, wheel, virtualenv
```
python -m pip install --upgrade pip wheel setuptools virtualenv	// install
python -m virtualenv [your virtual environment name]		// create virtual env. (you don't need to execute this one, the project already has a virtual env.)
[your virtual environment name]\Scripts\activate	// activate virtual env.
```
3. Install the dependencies for Kivy
```
python -m pip install docutils pygments pypiwin32 kivy_deps.sdl2==0.1.22 kivy_deps.glew==0.1.12
```
4. Install Kivy
```
python -m pip install kivy==1.11.1
```
5. Install application associated packages
```
python install tensorflow==1.14 regex
```

### Linux
[official guide from Kivy.org](https://kivy.org/doc/stable/installation/installation-linux.html)
1. Install Python 3.7
2. Setup pip, wheel, virtualenv (note that wheel in linux might run into issues)
```
python3 -m pip install --upgrade --user pip setuptools virtualenv
python3 -m virtualenv ~/kivy_venv   // create virtual env in home directory
source ~/kivy_venv/bin/activate	// activate virtual env.
```
3. Install Kivy
```
pip3 install kivy
```
4. Install application associated packages
```
pip3 install tensorflow==1.14 regex
```

## Language model setup
1. [GPT-2 models download](https://github.com/ConnorJL/GPT2)
enter GPT2_models/ and type:
```
pip3 install requests tqdm  // for execute download.py
python3 download_model.py PrettyBig
```