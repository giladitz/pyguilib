from tkinter import *
from random import randint


class Win(Frame):

    GREETING_MESSAGE = "Hello, and welcome!\n"

    def __init__(self, master=None, blob_file=None):
        Frame.__init__(self, master)
        self.master = master
        self.scroller, self.chat_box, self.response_box = None, None, None
        self.submit_button = None
        self.last_sentence = ""
        self.init_win()
        self.blob = ['It is true but not in the north.']
        if blob_file is not None:
            with open(blob_file, 'r') as fh:
                data = fh.read()
                self.blob += data.split('\n')

    def init_win(self):
        self.master.title("~ chatbot ~")
        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)
        self.scroller = Scrollbar(self)
        self.chat_box = Text(self, height=32, width=45, borderwidth="0")
        self.response_box = Text(self, height=3, width=36, borderwidth="0")
        # creating a button instance
        self.submit_button = Button(self, text="Submit",
                               command=self.send_msg, bg="lightblue",
                               borderwidth="0", highlightcolor="black",
                               height="3", width="10")
        """
        activebackground, activeforeground, anchor,
        background, bitmap, borderwidth, cursor,
        disabledforeground, font, foreground
        highlightbackground, highlightcolor,
        highlightthickness, image, justify,
        padx, pady, relief, repeatdelay,
        repeatinterval, takefocus, text,
        textvariable, underline, wraplength
        """
        self.scroller.pack(side=RIGHT, fill=Y)
        self.scroller.config(command=self.chat_box.yview)
        self.chat_box.place(x=1, y=1)
        self.response_box.place(x=1, y=540)
        self.submit_button.place(x=310, y=540)
        self.chat_box.insert(END, ">> {}".format(self.GREETING_MESSAGE))

        # configs
        self.chat_box.tag_configure('bot',
                                    foreground='#476042',
                                    font=('Tempus Sans ITC', 12, 'bold'))
        self.chat_box.tag_configure('user',
                                    foreground='#3953FA',
                                    font=('Tempus Sans ITC', 12, 'bold'))

        self.response_box.bind("<Key>", self.enter_key)

    def enter_key(self, event):
        if event.char == '\r':
            self.send_msg()

    def send_msg(self):
        # add msg to main bot window
        self.last_sentence = self.response_box.get("1.0", END)
        if self.last_sentence != "\n":
            self.chat_box.insert(END, ">> {}".format(self.last_sentence), 'user')
            self.response_box.delete("1.0", END)
            self.call_responder(randone=True)
            self.chat_box.see(END)

    def update_response(self, response):
        self.chat_box.insert(END, ">> {}\n".format(response.strip()), 'bot')

    def call_responder(self, randone=False):
        if randone:
            resp = self.blob[randint(0, len(self.blob)-1)]
        else:
            pass
            #resp = apicall-to-model
        self.update_response(resp)


root = Tk()
root.geometry("400x600")
app = Win(root, blob_file='blob')
root.mainloop()
