#
# Created on Mon Dec 18 2023
#
# Copyright (c) 2023 CEA - LASTI
# Contact: Evan Dufraisse,  evan[dot]dufraisse[at]cea[dot]fr. All rights reserved.
#
import fasttext


class FastLanguageIdentification:
    def __init__(self, path_language_model):
        pretrained_lang_model = path_language_model
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(
            text.replace("\n", " "), k=1
        )  # returns top 1 matching languages
        return predictions[0][0].split("__label__")[1], predictions[1][0]
