from fastai.vision.all import *
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

def cat_breeds(x): return x[0].isupper()

learn = load_learner('cat-model.pkl')

categories = ('Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Egyptian Mau', 'Maine Coon', 'Persian', 'Ragdoll', 'Russian Blue', 'Siamese', 'Sphynx')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(224,224))
label = gr.outputs.Label()
examples = ['cat01.jpg', 'cat02.jpg', 'cat03.jpg', 'cat04.jpg', 'cat05.jpg', 'cat1.jpg', 'cat2.jpg', 'cat.jpg']
thumbnail= ['thumbnail.jpg']

intf = gr.Interface(fn=classify_image, 
                    inputs=image, 
                    outputs=label, 
                    examples=examples, 
                    thumbnail=thumbnail,
                    title="Cat Breeds Classifier",
                    description = "<h3 style='font-weight=500'>At this time, this app can successfully identify 12 different types of cat bread: Abyssinian, Bengal, Birman, Bombay, British, Shorthair, Egyptian Mau, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese and Sphynx </h3>",
                    article="<p style='text-align: center'>Made with ❤️ by <a href='https://twitter.com/imbikramsaha/' target='_blank' style='color=blue'>Bikram Saha</a></p>"
                   )
intf.launch(inline=False)