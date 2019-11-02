import kivy
kivy.require('1.11.1')

from kivy.app import App
from kivy.uix.label import Label
from gpt2_pirate import Our_gpt2


class MyApp(App):

    def build(self):
        #message = Our_gpt2("GPT2_models/PrettyBig.json", None, "Daniel Han has just become the new predident of Taiwan.", 40)
        return Label(text="This fucking message about fucking min shu is the following.\n" + message)


if __name__ == '__main__':
    MyApp().run()
