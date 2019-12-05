from flask import request, url_for
from flask_api import FlaskAPI, status, exceptions
import numpy as np
import pandas as pd
import ufc_utils
import ufc_logistic_regression
import json

app = FlaskAPI(__name__)


@app.route('/logistic_regression')
def example():
    raw_frame = ufc_logistic_regression.ufc_data
    raw_frame = pd.DataFrame(np.zeros(shape=(1, raw_frame.shape[1])), columns=raw_frame.columns)
    blue = ''
    red = ''
    for fighter, stats in request.data.items():
        if blue == '':
            blue = fighter
        else:
            red = fighter

        for stat, value in stats.items():
            if stat in raw_frame.columns:
                raw_frame[stat][0] = value
    fighter_diffs = ufc_utils.create_diffs(raw_frame)
    fighter_diffs = fighter_diffs.fillna(0)
    prediction = ufc_logistic_regression.model.predict(fighter_diffs.drop('BlueFightWin', axis=1))
    if prediction[0] == 0:
        winner = blue
    else:
        winner = red
    return {"Winner": str(winner)}


if __name__ == "__main__":
    app.run(debug=True)
