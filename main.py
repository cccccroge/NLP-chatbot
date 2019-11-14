import kivy
kivy.require('1.11.1')

from kivy.app import App
from kivy.config import Config
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.layout import Layout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from gpt2_pirate import Our_gpt2


class NLPBotApp(App):

    def build(self):
        #message = Our_gpt2("GPT2_models/PrettyBig.json", None, "Daniel Han has just become the new predident of Taiwan.", 40)
        #return Label(text="This fucking message about fucking min shu is the following.\n" + message)
        return MainWindow()


class MainWindow(Widget):
    pass

if __name__ == '__main__':
    Config.set('graphics', 'fullsreen', 1)
    Config.set('graphics', 'window_state', 'maximized')
    Config.write()
    NLPBotApp().run()
