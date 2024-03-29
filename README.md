# NLP Bot
An easy NLP application running on Windows, using state-of-art deep learning models.
![demo_image](/images/demo.png)

### 功能
- 故事生成：選擇生成故事風-->輸入開頭文字-->生成該風格故事段落
- 情感分析：輸入文字-->回覆可能之情緒（回覆一至兩種，共13種）
- 經驗猜測：輸入文字-->產生數個可能猜測-->使用者給予回饋（是/否）

### 架構
![func_1](images/gen_story.jpg)
![func_2](images/emotion_class.jpg)
![func_3](images/experience_guess.jpg)

### 注意
實作上目前僅提供local端的CPU運算，第一個及第三個功能將會花費5分鐘以上的時間

# Project Development
## Installation
### Windows
[official guide from Kivy.org](https://kivy.org/doc/stable/installation/installation-windows.html)
1. Install Python 3.7
2. Setup pip, wheel, virtualenv
```
python -m pip install --upgrade pip wheel setuptools virtualenv	// install
python -m virtualenv [your virtual environment name]		// create virtual env. (you don't need to execute this one, the project already has a virtual env.)
source [your virtual environment name]\Scripts\activate	// activate virtual env.
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
python -m pip install tensorflow==1.14 regex
```

### Linux
[official guide from Kivy.org](https://kivy.org/doc/stable/installation/installation-linux.html)
1. Install Python 3.7 and pip
```
sudo apt-get update
sudo apt-get install python3-pip

```
2. Setup pip, wheel, virtualenv (note that wheel in linux might run into issues)
```
python3 -m pip install --upgrade --user pip setuptools virtualenv
python3 -m virtualenv ./[your_worn_venv]   // create virtual env at project root
source [your_worn_venv]/bin/activate	// activate virtual env.
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
1. for story telling
```
pip install gpt-2-simple
```
2. for emotion classification
```
pip install bert-tensorflow
pip install tensorflow_hub
pip install pandas
```
3. for experience guessing
```
pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch_pretrained_bert
pip install nltk
```