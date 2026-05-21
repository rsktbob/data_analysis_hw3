# Data Analysis Homework 3

本專案為資料探勘作業三的模型實作，使用 `train.csv` 訓練二元分類模型，並對 `test.csv` 產生可提交的預測結果。  
目前程式比較了 Logistic Regression、Random Forest 與 Gradient Boosting 三類模型，並以交叉驗證的 AUC 評估模型表現。

## 檔案介紹

### 資料檔案

| 檔案 | 說明 |
| --- | --- |
| `train.csv` | 訓練資料，包含目標欄位 `ACTION`。 |
| `test.csv` | 測試資料，不包含 `ACTION`，用來產生預測結果。 |
| `sampleSubmission.csv` | submission 格式範例，程式會依照此格式輸出預測檔。 |

### 程式檔案

| 檔案 | 說明 |
| --- | --- |
| `preprocessing.py` | 共用工具程式，負責讀取資料、列印資料摘要、Out-of-Fold target encoding、交叉驗證 AUC 評估，以及儲存 submission。 |
| `logistic_v1.py` | Logistic Regression 模型，先對類別特徵做 target encoding，再以標準化後的特徵訓練模型。 |
| `logistic_v1_1.py` | Logistic Regression 的 one-hot encoding 版本，直接將類別欄位展開後訓練模型。 |
| `random_forest_v2.py` | Random Forest 模型，使用 target encoding 後的特徵訓練。 |
| `random_forest_v2_1.py` | Random Forest 調參版本，增加樹的數量並調整樹深與葉節點限制。 |
| `gradient_boosting_v3.py` | Gradient Boosting 模型，使用 target encoding 後的特徵訓練。 |
| `gradient_boosting_v3_1.py` | Gradient Boosting 調參版本，調整估計器數量、學習率與葉節點限制。 |

### 輸出與環境檔案

| 檔案或資料夾 | 說明 |
| --- | --- |
| `submissions/` | 存放各模型產生的 submission CSV 檔案。 |
| `requirements.txt` | Python 套件需求，包含 `pandas`、`numpy` 與 `scikit-learn`。 |
| `DM_Homework_3.pdf` | 作業題目說明文件。 |
| `資料探勘hw3.pdf` | 本作業相關 PDF 文件。 |

## 主要特徵

模型使用下列類別特徵進行訓練：

- `RESOURCE`
- `MGR_ID`
- `ROLE_ROLLUP_1`
- `ROLE_ROLLUP_2`
- `ROLE_DEPTNAME`
- `ROLE_TITLE`
- `ROLE_FAMILY_DESC`
- `ROLE_FAMILY`
- `ROLE_CODE`

目標欄位為 `ACTION`，submission 會以測試資料的 `id` 對應預測機率。

## 安裝套件

```powershell
pip install -r requirements.txt
```

## 執行方式

在專案根目錄執行任一模型腳本，例如：

```powershell
python gradient_boosting_v3.py
```

也可以指定交叉驗證 fold 數、target encoding smoothing 參數與亂數種子：

```powershell
python gradient_boosting_v3.py --folds 5 --smoothing 10 --seed 42
```

one-hot Logistic Regression 版本沒有 `--smoothing` 參數：

```powershell
python logistic_v1_1.py --folds 5 --seed 42
```

程式執行後會：

1. 載入訓練資料、測試資料與 submission 範例。
2. 顯示資料摘要與特徵唯一值數量。
3. 以 Stratified K-Fold 進行交叉驗證並輸出 AUC。
4. 使用完整訓練資料訓練模型。
5. 將預測結果輸出到 `submissions/`。

## Submission 檔案

目前各模型輸出檔案如下：

| 模型腳本 | 輸出檔案 |
| --- | --- |
| `logistic_v1.py` | `submissions/submission_1_target_encoding_logistic.csv` |
| `logistic_v1_1.py` | `submissions/submission_1_1_one_hot_logistic.csv` |
| `random_forest_v2.py` | `submissions/submission_2_target_encoding_random_forest.csv` |
| `random_forest_v2_1.py` | `submissions/submission_2_1_target_encoding_random_forest.csv` |
| `gradient_boosting_v3.py` | `submissions/submission_3_target_encoding_gradient_boosting.csv` |
| `gradient_boosting_v3_1.py` | `submissions/submission_3_1_tuned_gradient_boosting.csv` |
