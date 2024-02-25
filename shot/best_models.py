import os
import pickle

best = {}

if __name__ == "__main__":
    """
    Returns information about the models and their performance. Used for writing up
    perf in report.
    """
    models_folder = 'shot/models'
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
                if test_score < best_score:
                    best[version] = (test_score, model_obj)

    for version in best:
        score, model = best[version]
        print(f"{version}: {score} - {model.score} - {model.model}")
