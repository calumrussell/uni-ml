from v1 import V1
from v2 import V2
from v3 import V3
from v4 import V4

if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore")

    v1 = V1.train()
    print(v1.score)

    v2 = V2.train()
    print(v2.score)

    v3 = V3.train()
    print(v3.score)

    v4 = V4.train()
    print(v4.score)
