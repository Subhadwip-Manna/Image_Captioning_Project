from flask import Flask, render_template, request
import pickle
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__, static_folder='Flicker8k_Dataset')

model = load_model('model50.h5')
features = pickle.load(open('features.pkl', 'rb'))
all_caption = pickle.load(open('all_captions.pkl', 'rb'))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_caption)

max_caption_length = 34


# Get the word corresponding to an index from the tokenizer's word index
def get_word_from_index(index):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None


def predict_caption(model, image_features):
    caption = 'startseq'
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        predicted_word = get_word_from_index(np.argmax(yhat))
        caption += " " + predicted_word
        if predicted_word is None or predicted_word == 'endseq':
            break
    return caption


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files['image']
        # Get the image name without the ".jpg" extension
        image_name = image.filename.split('.')[0]

        # Access the corresponding image features using the dictionary
        image_features = features.get(image_name)

        if image_features is None:
            caption = " Please take image from Flicker8k_Dataset directory"

        else:
            caption = predict_caption(model, image_features)
            caption = " ".join(caption.split(" ")[1:-1])

        image_url = f'/Flicker8k_Dataset/{image_name}.jpg'

        print("Image URL:", image_url)  # Debug statement

        return render_template('index.html', caption=caption, image_url=image_url)

    return render_template('index.html', caption=None)


if __name__ == '__main__':
    app.run(debug=True)
