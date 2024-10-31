import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.preprocessing import FunctionTransformer

#import custom features getting functions
from Features.drop_some_columns import condense_columns
from Features.get_player_avg_crossing import get_team_total_crossings
from Features.get_player_avg_rating import add_team_avg_rating
from Features.get_player_work_rate import add_player_work_rate_columns




def __main__():
    #get required data
    test = pd.read_csv("data/test.csv")
    games = pd.read_csv("data/games.csv")
    players = pd.read_csv("data/players.csv")
    
    #only keep non draw score
    games = games[games["home_team_goal"] != games["away_team_goal"]  ]
    
    games["winner"] = np.where((games["home_team_goal"] - games["away_team_goal"]) > 0, 1,0 )
    games["winby"] = abs(games["home_team_goal"] - games["away_team_goal"])


    # First predicting winbys

    #removing an outlier, there was only 2 games or so with a winby of 9 or higher
    games = games[(games["winby"] < 9)]

    # Define features and target for calculating winby
    X = games.drop(columns=["winby", "home_team_goal", "away_team_goal", "year","winner"])  # drop vars we cant use for predictions
    y = games["winby"] 

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)

    # win by model
    model_winby = make_pipeline(
        FunctionTransformer(add_player_work_rate_columns, kw_args={"players_df": players}),
        FunctionTransformer(add_team_avg_rating, kw_args={"players_df": players}),
        FunctionTransformer(get_team_total_crossings, kw_args={"players_df": players}),
        FunctionTransformer(condense_columns),
        SimpleImputer(strategy="mean"),
        XGBRegressor(
            n_estimators=200,         # Number of trees in the ensemble
            learning_rate=0.05,       # Smaller step size, requires more trees
            max_depth=5,              # Balanced tree depth for complexity
            subsample=0.8,            # Ratio of samples used per tree
            colsample_bytree=0.7,     # Fraction of features used per tree
            gamma=0.1,                # Minimum loss reduction required to make a further partition
            random_state=42           # Ensures reproducibility))
        )
    )


    # Fit the model using the resampled data and sample weights
    model_winby.fit(X_train, y_train)


    #test model_winby using X_test data
    print("Model_winby test data score: ",model_winby.score(X_test,y_test))

    #predict winby for test dataset
    win_by_test_prediction = model_winby.predict(test)
 

    # Now predicting winner

    X = games.drop(columns=["winby", "home_team_goal", "away_team_goal", "year", "winner"])
    y = games["winner"] 

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # winner model
    model_winner = make_pipeline(
        FunctionTransformer(add_player_work_rate_columns, kw_args={"players_df": players}),
        FunctionTransformer(add_team_avg_rating, kw_args={"players_df": players}),
        FunctionTransformer(get_team_total_crossings, kw_args={"players_df": players}),
        FunctionTransformer(condense_columns),
        SimpleImputer(strategy="mean"),
        XGBRegressor(
            n_estimators=200,         # Number of trees in the ensemble
            learning_rate=0.05,       # Smaller step size, requires more trees
            max_depth=5,              # Balanced tree depth for complexity
            subsample=0.8,            # Ratio of samples used per tree
            colsample_bytree=0.7,     # Fraction of features used per tree
            gamma=0.1,                # Minimum loss reduction required to make a further partition
            random_state=42           # Ensures reproducibility))
        )
    )

    model_winner.fit(X_train, y_train)

    print("Model_winner score:", model_winner.score(X_test, y_test))

    # df of winner and winby predictions for the given test dataset
    prediction_for_test = pd.DataFrame({'winby': (win_by_test_prediction), "winner": model_winner.predict(test)})

    # Save the DataFrame to CSV
    prediction_for_test[["winner","winby"]].to_csv("Output/pred_for_test.csv", index=None)


if __name__ == "__main__":
    __main__()


