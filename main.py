import kivy
kivy.require('1.11.1')

from kivy.app import App
from kivy.config import Config
from kivy.core.text import LabelBase
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, OptionProperty, DictProperty
from functions import gen_story
from message import msg
from font import fonts
from output_format import wrap_text_length

import time
from threading import Thread


class NLPBotApp(App):

    def build(self):
        return MainWindow()


class MainWindow(Widget):
    # UI stuff
    top_layout = ObjectProperty(None)
    response_label = ObjectProperty(None)
    output_label = ObjectProperty(None)
    output_layout = ObjectProperty(None)
    confirm_layout = ObjectProperty(None)

    bottom_layout = ObjectProperty(None)
    functions_layout = ObjectProperty(None)
    input_layout = ObjectProperty(None)
    text_input = ObjectProperty(None)
    ask_layout = ObjectProperty(None)
    ask_label = ObjectProperty(None)
    
    message = DictProperty(msg)

    # Function state
    state = OptionProperty("NONE", options=["NONE", "WAIT_SERVE", "HINT_SHOWN", "SELECT_FUNC", \
    "ENTER_INPUT", "WAIT_OUTPUT", "OUTPUT_SHOWN"])
    

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hideAll()
        self.state = "WAIT_SERVE"
        self.current_func = None

    def hideAll(self):
        self.top_layout.remove_widget(self.response_label)
        self.top_layout.remove_widget(self.output_layout)
        self.top_layout.remove_widget(self.confirm_layout)

        self.toggleInterface('input', False)
        self.toggleInterface('functions', False)
        self.toggleInterface('ask', False)

    def toggleInterface(self, which, on):
        if (which == 'response'):
            if (on):
                self.top_layout.add_widget(self.response_label)
                self.top_layout.size_hint_y = 0.5
            else:
                self.top_layout.remove_widget(self.response_label)

        elif (which == 'output'):
            if (on):
                self.top_layout.add_widget(self.output_layout)
                self.top_layout.add_widget(self.confirm_layout)
                self.top_layout.size_hint_y = 1.0
            else:
                self.top_layout.remove_widget(self.output_layout)
                self.top_layout.remove_widget(self.confirm_layout)
        
        elif (which == 'input'):
            if (on):
                self.bottom_layout.add_widget(self.input_layout)
            else:
                self.bottom_layout.remove_widget(self.input_layout)

        elif (which == 'functions'):
            if (on):
                self.bottom_layout.add_widget(self.functions_layout)
            else:
                self.bottom_layout.remove_widget(self.functions_layout)

        elif (which == 'ask'):
            if (on):
                self.bottom_layout.add_widget(self.ask_layout)
            else:
                self.bottom_layout.remove_widget(self.ask_layout)

    def on_click_bot(self):
        if (self.state == "WAIT_SERVE"):
            self.state = "HINT_SHOWN"
            self.toggleInterface('response', True)
            self.response_label.text = self.message['welcome']

    def on_click_hint(self):
        if (self.state == "HINT_SHOWN"):
            self.state = "SELECT_FUNC"
            self.toggleInterface('response', False)
            self.toggleInterface('functions', True)

    def on_click_a(self):
        if (self.state == "SELECT_FUNC"):
            self.state = "ENTER_INPUT"
            self.current_func = 'a'
            self.toggleInterface('functions', False)
            self.toggleInterface('input', True)
            self.toggleInterface('response', True)
            self.response_label.text = self.message['func_a_hint']

    def on_click_b(self):
        if (self.state == "SELECT_FUNC"):
            self.state = "ENTER_INPUT"
            self.current_func = 'b'
            self.toggleInterface('functions', False)
            self.toggleInterface('input', True)
            self.toggleInterface('response', True)
            self.response_label.text = self.message['func_b_hint']

    def on_click_c(self):
        if (self.state == "SELECT_FUNC"):
            self.state = "ENTER_INPUT"
            self.current_func = 'c'
            self.toggleInterface('functions', False)
            self.toggleInterface('input', True)
            self.toggleInterface('response', True)
            self.response_label.text = self.message['func_c_hint']
    
    def on_click_say(self):
        if (self.state == 'ENTER_INPUT'):
            self.state = "WAIT_OUTPUT"
            self.response_label.text = self.message['calculate']
            self.toggleInterface('input', False)
            self.toggleInterface('ask', True)
            self.ask_label.text = self.text_input.text
            self.text_input.text = ''

            if (self.current_func == 'a'):
                # GPT-2
                thread_func_a = Thread(
                    target=self.func_a_concurrent,
                    args=("GPT2_models/PrettyBig.json", None, self.ask_label.text, 40, ))
                thread_func_a.start()
                
            elif (self.current_func == 'b'):
                pass
            elif (self.current_func == 'c'):
                pass

    def func_a_concurrent(self, json, text_file, text_str, topk):
        self.output_label.text = wrap_text_length(
            gen_story('sherlock', text_str), 50)
        self.state = "OUTPUT_SHOWN"
        self.toggleInterface('response', False)
        self.toggleInterface('output', True)


    def on_click_confirm(self):
        if (self.state == 'OUTPUT_SHOWN'):
            self.state = 'WAIT_SERVE'
            self.toggleInterface('ask', False)
            self.toggleInterface('output', False)
                


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
