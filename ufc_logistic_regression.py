from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import ufc_utils


def run_model(data):
    differences = ufc_utils.create_diffs(data)

    x_train, x_test, y_train, y_test = train_test_split(
        differences.drop('BlueFightWin', axis=1),
        differences['BlueFightWin'],
        test_size=0.30,
        random_state=1
    )

    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model


