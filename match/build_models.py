from v1 import V1
from v2 import V2
from v3 import V3
from v4 import V4
from v5 import V5

if __name__ == "__main__":

    windows = [10, 20, 30, 40, 50, 75, 100]
    for window in windows:
        print(window)
        v1 = V1.train(window)
        print(v1.score)

        v2 = V2.train(window)
        print(v2.score)

        v3 = V3.train(window)
        print(v3.score)

        v4 = V4.train(window)
        print(v4.score)

        v5= V5.train(window)
        print(v5.score)
