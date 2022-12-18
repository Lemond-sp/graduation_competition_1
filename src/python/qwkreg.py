import numbers

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score


def lgb_custom_metric_qwk_regression(preds, data):
    """LightGBM のカスタムメトリックを計算する関数

    回帰問題として解いた予測から QWK を計算する"""
    # 正解ラベル
    y_true = data.get_label()
    # 予測ラベル
    y_pred = np.clip(preds, 0, 7).round()  # 単純に予測値を clip して四捨五入する
    # QWK を計算する
    return 'qwk', qwk(y_true, y_pred), True


def qwk(y_true, y_pred):
    """QWK (Quadratic Weighted Kappa) を計算する関数"""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def _numerical_only_df(df_):
    """数値のカラムだけ取り出す関数"""
    number_cols = [name for name, dtype in df_.dtypes.items()
                   if issubclass(dtype.type, numbers.Number)]
    numerical_df = df_[number_cols]
    return numerical_df


def main():
    # データセットを読み込む
    train_df = pd.read_csv('train.csv.zip')

    # 説明変数
    x_train = train_df.drop('Response', axis=1)
    # 簡単にするため、とりあえずすぐに使えるものだけ取り出して使う
    x_train = _numerical_only_df(x_train)

    # 目的変数
    y_train = train_df.Response.astype(float)
    # 目的変数はゼロからスタートにする
    y_train -= 1

    # LightGBM のデータセット表現にする
    lgb_train = lgb.Dataset(x_train, y_train)

    # 回帰問題として解く
    lgbm_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'first_metric_only': True,
        'verbose': -1,
    }

    # データ分割の方法
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 5-Fold CV で検証する
    result = lgb.cv(lgbm_params,
                    lgb_train,
                    num_boost_round=1000,
                    early_stopping_rounds=100,
                    folds=folds,
                    verbose_eval=10,
                    feval=lgb_custom_metric_qwk_regression,
                    seed=42,
                    )

    # early stop したラウンドでの QWK を出力する
    print(f'CV Mean QWK: {result["qwk-mean"][-1]}')


if __name__ == '__main__':
    main()