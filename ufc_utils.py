import pandas as pd


def acceptable_inputs(data_frame):
    columns = data_frame.columns.to_list()
    attrs = {}
    for column in columns:
        if column[:2] == 'B_':
            if 'R_' + column[2:] in columns:
                columns.remove('R_' + column[2:])
                attrs[column[2:]] = [column[2:].replace('_', ' ').upper(), column, 'R_' + column[2:]]
        elif column[:2] == 'R_':
            if 'B_' + column[2:] in columns:
                columns.remove('B_' + column[2:])
                attrs[column[2:]] = [column[2:].replace('_', ' ').upper(), 'B_' + column[2:], column]
    return attrs


def create_diffs(ufc):
    diffs = pd.DataFrame(
        data={
            "BlueFightWin": ufc['Winner'].replace('Blue', 1).replace('Red', 0),
            "loseSteakDiff": ufc['B_current_lose_streak'] - ufc['R_current_lose_streak'],
            "koDiff": ufc['B_win_by_KO/TKO'] - ufc['R_win_by_KO/TKO'],
            "submissionDiff": ufc['B_win_by_Submission'] - ufc['R_win_by_Submission'],
            "doctorStoppageWinDiff": ufc['B_win_by_TKO_Doctor_Stoppage'] - ufc['R_win_by_TKO_Doctor_Stoppage'],
            "winDiff": ufc['B_wins'] - ufc['R_wins'],
            # Decision Tree probit model value for stance vs. stance
            "reachDiff": ufc['B_Reach_cms'] - ufc['R_Reach_cms'],
            "heightDiff": ufc['B_Height_cms'] - ufc['R_Height_cms'],
            "weightDiff": ufc['B_Weight_lbs'] - ufc['R_Weight_lbs'],
            "ageDiff": ufc['B_age'] - ufc['R_age'],
            # Likely hood of fighter win diff by number of rounds
            "avgBodyAttDiff": ufc['B_avg_BODY_att'] - ufc['R_avg_BODY_att'],
            "avgBodyLandDiff": ufc['B_avg_BODY_landed'] - ufc['R_avg_BODY_landed'],
            "avgBodyLandPercDiff": (ufc['B_avg_BODY_landed'] / ufc['B_avg_BODY_att']) -
                                   (ufc['R_avg_BODY_landed'] / ufc['R_avg_BODY_att']),
            "avgClinchAttDiff": ufc['B_avg_CLINCH_att'] - ufc['R_avg_CLINCH_att'],
            "avgClinchLandDiff": ufc['B_avg_CLINCH_landed'] - ufc['R_avg_CLINCH_landed'],
            "avgClinchLandPercDiff": (ufc['B_avg_CLINCH_landed'] / ufc['B_avg_CLINCH_att']) -
                                     (ufc['R_avg_CLINCH_landed'] / ufc['R_avg_CLINCH_att']),
            "avgDistanceAttDiff": ufc['B_avg_DISTANCE_att'] - ufc['R_avg_DISTANCE_att'],
            "avgDistanceLandDiff": ufc['B_avg_DISTANCE_landed'] - ufc['R_avg_DISTANCE_landed'],
            "avgDistanceLandPercDiff": (ufc['B_avg_DISTANCE_landed'] / ufc['B_avg_DISTANCE_att']) -
                                       (ufc['R_avg_DISTANCE_landed'] / ufc['R_avg_DISTANCE_att']),
            "avgGroundAttDiff": ufc['B_avg_GROUND_att'] - ufc['R_avg_GROUND_att'],
            "avgGroundLandDiff": ufc['B_avg_GROUND_landed'] - ufc['R_avg_GROUND_landed'],
            "avgGroundLandPercDiff": (ufc['B_avg_GROUND_landed'] / ufc['B_avg_GROUND_att']) -
                                     (ufc['R_avg_GROUND_landed'] / ufc['R_avg_GROUND_att']),
            "avgHeadAttDiff": ufc['B_avg_HEAD_att'] - ufc['R_avg_HEAD_att'],
            "avgHeadLandDiff": ufc['B_avg_HEAD_landed'] - ufc['R_avg_HEAD_landed'],
            "avgHeadLandPercDiff": (ufc['B_avg_HEAD_landed'] / ufc['B_avg_HEAD_att']) -
                                   (ufc['R_avg_HEAD_landed'] / ufc['R_avg_HEAD_att']),
            "avgLegAttDiff": ufc['B_avg_LEG_att'] - ufc['R_avg_LEG_att'],
            "avgLegLandDiff": ufc['B_avg_LEG_landed'] - ufc['R_avg_LEG_landed'],
            "avgLegLandPercDiff": (ufc['B_avg_LEG_landed'] / ufc['B_avg_LEG_att']) -
                                  (ufc['R_avg_LEG_landed'] / ufc['R_avg_LEG_att']),
            "avgKnockDownDiff": ufc['B_avg_KD'] - ufc['R_avg_KD'],
            "avgPassDiff": ufc['B_avg_PASS'] - ufc['R_avg_PASS'],
            "avgRevDiff": ufc['B_avg_REV'] - ufc['B_avg_REV']
        }
    )
    diffs.drop('BlueFightWin', axis=1).head(5)
    # sns.heatmap(diffs.isnull(), yticklabels=False, cmap='viridis')

    diffs = diffs.fillna(diffs.mean())
    # sns.heatmap(diffs.isnull(), yticklabels=False, cmap='viridis')
    diffs = diffs[diffs['BlueFightWin'] != 'Draw']
    diffs = diffs.astype({'BlueFightWin': 'int32'})

    return diffs
