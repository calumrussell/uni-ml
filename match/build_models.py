from v1 import V1PoissonToLogisticProbability, V1PoissonToPoissonProbability
from v2 import V2PoissonToLogisticProbability

if __name__ == "__main__":

    windows = [10, 20, 30, 40, 50, 75, 100]
    for window in windows:
        print(window)
        v1_logistic = V1PoissonToLogisticProbability.train(window)
        print(v1_logistic.score)
        v1_poisson = V1PoissonToPoissonProbability.train(window)
        print(v1_poisson.score)
        v2 = V2PoissonToLogisticProbability.train(window)
        print(v2.score)
