# NLP_application
An easy NLP application running on Android system, mainly focus on helping students to write their life diary in a fun way.

# Project Development
## Windows
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
