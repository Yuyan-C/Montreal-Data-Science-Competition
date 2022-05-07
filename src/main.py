
from model import *

if __name__ == "__main__":
    train_score, test_score, result = train_and_predict(model="rf", encode="label", na="zero")
    val = pd.read_csv("../data/validation.csv")
    print("train score = ", train_score)
    print("test score = ", test_score)
    output = pd.DataFrame(
        {'ID': val['unique_id'],
         'Prediction': result.tolist()
         }
    )
    output.to_csv("output.csv", index=False)

    count = 0
    for x in result:
        if x > 0.5:
            count += 1
    print(count)
