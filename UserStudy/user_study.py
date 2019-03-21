from tkinter import *
from tkinter import ttk
from random import shuffle
from PIL import ImageTk, Image
from tkinter.ttk import Progressbar

window = Tk()
window.title("User Study")
window.geometry('1024x512')



class ImageSet():
    def __init__(self, path, model, num):
        self.A = "%s/%d/%d/%s.png" % (path, model, num, "a")
        self.B = "%s/%d/%d/%s.png" % (path, model, num, "b")
        self.AtoB = "%s/%d/%d/%s.png" % (path, model, num, "ab")
        self.model = model

class GUI():

    def create_image_sets(self, path):
        image_sets = []
        for i in range(self.num_of_images // 2):
            image_sets += [ImageSet(path, 1, i)]
            image_sets += [ImageSet(path, 2, i)]
        shuffle(image_sets)
        return image_sets

    def __init__(self, root):
        self.root = root
        self.curr_image_set = 0
        self.num_of_images = 10
        self.image_sets = self.create_image_sets('.')
        self.frame = ttk.Frame()
        self.frame.grid(column=0, row=0)
        self.answers = [[0,0,0],[0,0,0]]
        self.answered_q = [False, False, False]

        lbl = Label(self.frame, text=("\n\nYou are about to be presented %d sets of "
                                  "images.\n\nEach set will be comprised with a "
                                  "'source' image (domain A - beard) and a 'guide' "
                                  "image (domain B - smile).\n A third image will be"
                                  "shown as well, which is a transfer from image A "
                                  "to B (domainwise), i.e. removal of beard and "
                                  "addition of smile.\n\n You will be asked three "
                                  "questions:\n1) Was the beard removed\n2) Was the "
                                  "smile added?\n3) How much the translation is the "
                                  "identity translation (i.e. the face from the "
                                  "source was kept).\n\n"
                                  "The possible answers will be values from 1-5 "
                                  "where 1 indicated VERY BAD result, "
                                  "and 5 indicates a VERY GOOD result.\n\n"
                                  "You will have 20 seconds to answer each image "
                                  "set.\n" %
                                  self.num_of_images), font=("Arial", 12))
        lbl.grid(column=0, row=0)

        btn = Button(self.frame, text="Start",
                     command=self.show_next_image_sets)
        btn.grid(column=0, row=1)

    def show_next_image_sets(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

        if self.curr_image_set == self.num_of_images - 1:
            txt_file = open("res.txt", "w")
            txt_file.write("%d,%d,%d\n%d,%d,%d" % (self.answers[0][0],
                                                   self.answers[0][1],
                                                   self.answers[0][2],
                                                   self.answers[1][0],
                                                   self.answers[1][1],
                                                   self.answers[1][2]))
            txt_file.close()
            self.root.quit()

        image_set = self.image_sets[self.curr_image_set]
        self.curr_image_set += 1

        self.imgA = ImageTk.PhotoImage(Image.open(image_set.A))
        imgAlbl = Label(self.frame, image=self.imgA)
        imgAlbl.grid(row=1, column=1)
        self.imgB = ImageTk.PhotoImage(Image.open(image_set.B))
        imgBlbl = Label(self.frame, image=self.imgB)
        imgBlbl.grid(row=1, column=2)
        self.imgAB = ImageTk.PhotoImage(Image.open(image_set.AtoB))
        imgABlbl = Label(self.frame, image=self.imgAB)
        imgABlbl.grid(row=1, column=3)

        lbl = Label(self.frame, text="Domain A").grid(row=2, column=1)
        lbl = Label(self.frame, text="Domain B").grid(row=2, column=2)
        lbl = Label(self.frame, text="Translated Image").grid(row=2, column=3)

        self.bar = Progressbar(self.frame, length = 100, orient='vertical')
        self.bar.grid(row=1, column=4)
        self.progress_bar_timer(0)

        lbl = Label(self.frame, text="How did it manage to remove "
                                     "beard?").grid(row=3, column=1)
        btn = Button(self.frame, text="1", command=lambda :
        self.button_answer(image_set,0,1)).grid(row=4, column=1)
        btn = Button(self.frame, text="2", command=lambda :
        self.button_answer(image_set,0,2)).grid(row=5, column=1)
        btn = Button(self.frame, text="3", command=lambda :
        self.button_answer(image_set,0,3)).grid(row=6, column=1)
        btn = Button(self.frame, text="4", command=lambda :
        self.button_answer(image_set,0,4)).grid(row=7, column=1)
        btn = Button(self.frame, text="5", command=lambda :
        self.button_answer(image_set,0,5)).grid(row=8, column=1)

        lbl = Label(self.frame, text="How did it manage to add "
                                     "smile?").grid(row=3, column=2)
        btn = Button(self.frame, text="1", command=lambda :
        self.button_answer(image_set,1,1)).grid(row=4, column=2)
        btn = Button(self.frame, text="2", command=lambda :
        self.button_answer(image_set,1,2)).grid(row=5, column=2)
        btn = Button(self.frame, text="3", command=lambda :
        self.button_answer(image_set,1,3)).grid(row=6, column=2)
        btn = Button(self.frame, text="4", command=lambda :
        self.button_answer(image_set,1,4)).grid(row=7, column=2)
        btn = Button(self.frame, text="5", command=lambda :
        self.button_answer(image_set,1,5)).grid(row=8, column=2)

        lbl = Label(self.frame, text="How did it manage to keep "
                                     "identity?").grid(row=3, column=3)
        btn = Button(self.frame, text="1", command=lambda :
        self.button_answer(image_set,2,1)).grid(row=4, column=3)
        btn = Button(self.frame, text="2", command=lambda :
        self.button_answer(image_set,2,2)).grid(row=5, column=3)
        btn = Button(self.frame, text="3", command=lambda :
        self.button_answer(image_set,2,3)).grid(row=6, column=3)
        btn = Button(self.frame, text="4", command=lambda :
        self.button_answer(image_set,2,4)).grid(row=7, column=3)
        btn = Button(self.frame, text="5", command=lambda :
        self.button_answer(image_set,2,5)).grid(row=8, column=3)

    def progress_bar_timer(self, num):
        self.bar['value'] = (num * 10.0) / 2
        self.after = self.root.after(1000, lambda: self.progress_bar_timer(num+1))

    def button_answer(self, image_set, q, a):
        if self.answered_q[q] == False:
            self.answers[image_set.model - 1][q] += a
            print(self.answers)
            self.answered_q[q] = True

        if self.answered_q[0] == True and self.answered_q[1] == True and \
                self.answered_q[2] == True:
            self.answered_q = [False, False, False]
            self.root.after_cancel(self.after)
            self.show_next_image_sets()

app = GUI(window)

window.mainloop()







