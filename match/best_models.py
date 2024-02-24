import os
import pickle
import numpy as np

np.set_printoptions(suppress=True)

best = {}

if __name__ == "__main__":
    """
    Returns information about the models and their performance. Used for writing up
    perf in report.
    """
    models_folder = 'match/models'
    models = os.listdir(models_folder)
    for model in models:
        with open(f"{models_folder}/{model}", 'rb') as f:
            model_obj = pickle.load(f)
            path, ext = model.split(".")
            version, epoch = path.split("_")

            test_score = model_obj.test_score()
            if version not in best:
                best[version] = (test_score, model_obj)
            else:
                best_score, model_obj = best[version]
                if test_score > best_score:
                    best[version] = (test_score, model_obj)

    for version in best:
        score, model = best[version]
        if hasattr(model, 'model'):
            print(f"{version}: {score} - {model.score} - {model.ratings_window_length} - {model.model}")
        else:
            print(f"{version}: {score} - {model.score}")
