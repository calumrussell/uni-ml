from v1 import V1
from v2 import V2
from v3 import V3

if __name__ == "__main__":
    v1 = V1.train()
    print(v1.score)

    v2 = V2.train()
    print(v2.score)

    v3 = V3.train()
    print(v3.score)
