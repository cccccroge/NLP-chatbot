import kivy
kivy.require('1.11.1')

from kivy.app import App
from kivy.config import Config
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from gpt2_pirate import Our_gpt2


class NLPBotApp(App):

    def build(self):
        #message = Our_gpt2("GPT2_models/PrettyBig.json", None, "Daniel Han has just become the new predident of Taiwan.", 40)
        #return Label(text="This fucking message about fucking min shu is the following.\n" + message)
        return MainWindow()


class MainWindow(Widget):
    hint_layout = ObjectProperty(None)
    functions_layout = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hideAll()

    def hideAll(self):
        self.hint_layout.opacity = 0
        self.hint_layout.disabled = True
        self.functions_layout.opacity = 0
        self.functions_layout.disabled = True

    def on_click_bot(self):
        self.hint_layout.opacity = 1
        self.hint_layout.disabled = False

    def on_click_hint(self):
        self.hint_layout.opacity = 0
        self.hint_layout.disabled = True
        self.functions_layout.opacity = 1
        self.functions_layout.disabled = False



if __name__ == '__main__':
    Config.set('graphics', 'fullsreen', 1)
    Config.set('graphics', 'window_state', 'maximized')
    Config.write()
    NLPBotApp().run()
