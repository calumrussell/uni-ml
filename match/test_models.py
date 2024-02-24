import os
import pickle

if __name__ == "__main__":
    """
    Used for returning the test scores of all the models currently saved to file.
    """
    models_folder = 'match/models'
    models = os.listdir(models_folder)
    for model in models:
        with open(f"{models_folder}/{model}", 'rb') as f:
            model_obj = pickle.load(f)
            print(f"{model}: {model_obj.test_score()}")

