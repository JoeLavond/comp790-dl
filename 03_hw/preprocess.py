# packages
import numpy as np
import pandas as pd


def main():

    # import data
    train = pd.read_csv('./data/kaggle_house_pred_train.csv')
    train.drop('Id', axis=1, inplace=True)
    test = pd.read_csv('./data/kaggle_house_pred_test.csv')
    test_id = test.Id
    test.drop('Id', axis=1, inplace=True)

    # test set indicator
    train['test_ind'] = 0
    test['test_ind'] = 1

    # combine datasets for manipulation
    data = pd.concat((train, test))
    test_ind = data.test_ind

    # set aside response
    resp = data['SalePrice']
    data.drop(['test_ind', 'SalePrice'], axis=1, inplace=True)

    # --------------------------------------------------------------

    """
    LotFrontage = ft street along house
        no 0 values
        missing = 0?
    GarageYrBlot 
        missing = no garage? already included with categorical NA
        2207 entry error for 2007?
    Fill remaining missing with mean imputation
    """

    # imputation
    data.loc[:, 'LotFrontage'] = 0 
    data.loc[data.GarageYrBlt == 2207, 'GarageYrBlot'] = 2007 

    # remaining mean fill
    num_ind = data.dtypes != 'object'
    data.loc[:, num_ind] = data.loc[:, num_ind].fillna(
        data.loc[:, num_ind].mean()
    )

    # ------------------------------------------------------------

    # standardize numerical data
    data.loc[:, num_ind] = data.loc[:, num_ind].apply(
        lambda x: (x - x.mean()) / x.std(), axis=1
    )

    # dummy vars
    data = pd.get_dummies(data, dummy_na=True)

    # -----------------------------------------------------------

    # return to train/test
    train_x, train_y = data.loc[test_ind == 0, :], resp[test_ind == 0]
    test_x = data.loc[test_ind == 1, :]

    # save processed data
    np.save('./data/train_x.npy', train_x.to_numpy())
    np.save('./data/train_y.npy', train_y.to_numpy())
    np.save('./data/test_x.npy', test_x.to_numpy())
    np.save('./data/test_id.npy', test_id)

    return None


if __name__ == "__main__":
    main()

