import kivy
kivy.require('1.11.1')

from kivy.app import App
from kivy.config import Config
from kivy.core.text import LabelBase
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, OptionProperty, DictProperty
from gpt2_pirate import Our_gpt2
from message import msg
from font import fonts


class NLPBotApp(App):

    def build(self):
        #message = Our_gpt2("GPT2_models/PrettyBig.json", None, "Daniel Han has just become the new predident of Taiwan.", 40)
        #return Label(text="This fucking message about fucking min shu is the following.\n" + message)
        return MainWindow()


class MainWindow(Widget):
    output_layout = ObjectProperty(None)
    output_label = ObjectProperty(None)
    functions_layout = ObjectProperty(None)
    input_layout = ObjectProperty(None)
    bottom_layout = ObjectProperty(None)

    state = OptionProperty("NONE", options=["NONE", "WAIT_SERVE", "HINT_SHOWN", "SELECT_FUNC", \
    "ENTER_INPUT", "WAIT_RESPONSE", "OUTPUT_SHOWN"])
    message = DictProperty(msg)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hideAll()
        self.state = "WAIT_SERVE"

    def hideAll(self):
        self.output_layout.opacity = 0
        self.output_layout.disabled = True

        self.toggleLayout('input')
        self.toggleLayout('functions')


    def toggleLayout(self, which):
        if (which == 'output'):
            if (self.output_layout.disabled):
                self.output_layout.opacity = 1
                self.output_layout.disabled = False
            else:
                self.output_layout.opacity = 0
                self.output_layout.disabled = True
        elif (which == 'input'):
            if (self.state == 'ENTER_INPUT'):   # show input
                self.bottom_layout.add_widget(self.input_layout)
            else:
                self.bottom_layout.remove_widget(self.input_layout)

        elif (which == 'functions'):
            if (self.state == 'SELECT_FUNC'):   # show functions
                self.bottom_layout.add_widget(self.functions_layout)
            else:
                self.bottom_layout.remove_widget(self.functions_layout)

    def setRobotText(self, which):
        self.output_label.text = self.message[which]

    def on_click_bot(self):
        if (self.state == "WAIT_SERVE"):
            self.state = "HINT_SHOWN"
            self.toggleLayout('output')
            

    def on_click_hint(self):
        if (self.state == "HINT_SHOWN"):
            self.state = "SELECT_FUNC"
            self.toggleLayout('output')
            self.toggleLayout('functions')
    

    def on_click_a(self):
        if (self.state == "SELECT_FUNC"):
            self.state = "ENTER_INPUT"
            self.toggleLayout('functions')
            self.toggleLayout('input')
            self.toggleLayout('output')
            self.setRobotText('func_a_hint')

    def on_click_b(self):
        if (self.state == "SELECT_FUNC"):
            self.state = "ENTER_INPUT"
            self.toggleLayout('functions')
            self.toggleLayout('input')
            self.toggleLayout('output')
            self.setRobotText('func_b_hint')

    def on_click_c(self):
        if (self.state == "SELECT_FUNC"):
            self.state = "ENTER_INPUT"
            self.toggleLayout('functions')
            self.toggleLayout('input')
            self.toggleLayout('output')
            self.setRobotText('func_c_hint')
            

    # For debug
    def on_state(self, instance, value):
        print("state beocme: " + value)



if __name__ == '__main__':

    Config.set('graphics', 'fullsreen', 1)
    Config.set('graphics', 'window_state', 'maximized')
    Config.write()

    for font in fonts:
        LabelBase.register(**font)

    NLPBotApp().run()
