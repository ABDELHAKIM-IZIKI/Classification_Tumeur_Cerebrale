from joblib import load
from tkinter import Label , Canvas ,Button ,filedialog
from  tkdnd  import TkinterDnD , DND_FILES
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from PIL import Image, ImageTk

Categories = ['Gliome', 'Méningiome', 'No_tumeur', 'Pituitaire']
model = load('Tumeur_Cérébrale_KNN.joblib')

old_img = None


def prédire_tumeur(event):
    global old_img 

    
    
    path = event.data
    
    if path:
        img = imread(path)
        imag_resize1 = resize(img, (150, 150, 3))
        l = [imag_resize1.flatten()]

        type = Categories[model.predict(l)[0]]
        if type == 'No_tumeur':
            text_Label.config(text="L'image prédite est : No tumeur cérébrale", font=("Arial", 15, "bold"), background='#73C2FB')
        else:
            text_Label.config(text=f"L'image prédite est : Tumeur cérébrale de type {type}", font=("Arial", 15, "bold") , background='#73C2FB')
        
        



        img = rgb2gray(img)
        imag_resize2 = resize(img, (512,512))
        imag_resize2 = ImageTk.PhotoImage(Image.fromarray((imag_resize2*255).astype('uint8')))

        if old_img is not None:
            canvas.delete(old_img)
        old_img = canvas.create_image(0,0, anchor='nw' , image=imag_resize2)
        canvas.image = imag_resize2


root = TkinterDnD.Tk()
root.title("Détection de tumeurs cérébrales")
root.geometry("700x750")
root.configure(bg='#73C2FB')

drop_label = Label(root, text="Déposez l'image ici pour prédire le type de tumeur cérébrale", background="#003152", width=65, height=3, relief="ridge", font=("Arial", 13, "bold"), foreground="#FFFFFF")
drop_label.pack(pady=20)



drop_label.drop_target_register(DND_FILES)
drop_label.dnd_bind('<<Drop>>', prédire_tumeur)



text_Label = Label(root, text="", background='#73C2FB')
text_Label.pack(pady=10)

canvas = Canvas(root, width=512, height=512 , background='#73C2FB', relief='ridge')
canvas.pack()

root.mainloop()