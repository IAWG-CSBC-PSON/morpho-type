# morpho-type
Prediction of cell type from morphology features and neighborhood information

## Installation

```
git clone https://github.com/IAWG-CSBC-PSON/morpho-type.git
cd morpho-type
Rscript setup.R
```
This will clone the repository and install R packages needed to run the code.

## Data

Morphological features, neighborhood information and class labels for three lung cancer specimens reside in the `data/` subdirectory. The files are in .csv format containing 21 columns. The interpretation of columns is as follows:

  * `CellID` - a unique identifier of each cell within the specimen
  * `Label` - cell type label, computed as argmax of probabilities in `Immune`, `Stroma` and `Tumor` columns.
  * `Immune`, `Stroma`, `Tumor` - posterior probabilities of marker expression, computed by fitting a mixture of two Gaussians to CD45, alpha-SMA and Kertain respectively.
  * `Area` through `Orientation` - morphological features extracted from segmented cell populations
  * `Neighbor_*` - cell IDs of five immediate neighbors (the value of 0 is used as padding when a cell has fewer than five neighbors).
  * `X_posiiton` and `Y_position` - coordinates of the cell in the original raw image.

The files were wrangled based on data published in [Rashid, et al. (2019)](https://www.nature.com/articles/s41597-019-0332-y) and can be regenerated using `Rscript wrangle.R`.

### Raw images
Raw images from which all features were derived can be downloaded from Synapse together with their segmentation masks:

  * Lung1 -> [Image](https://www.synapse.org/#!Synapse:syn17774887), [Segmentation Mask](https://www.synapse.org/#!Synapse:syn21636326)
  * Lung2 -> [Image](https://www.synapse.org/#!Synapse:syn17776482), [Segmentation Mask](https://www.synapse.org/#!Synapse:syn21636350)
  * Lung3 -> [Image](https://www.synapse.org/#!Synapse:syn17778717), [Segmentation Mask](https://www.synapse.org/#!Synapse:syn21636364)

R code to download all of these at once is as follows:

``` R
synapser::synLogin()
syn <- synExtra::synDownloader("/path/to/data/folder/")
syn("syn17774887", "syn17776482", "syn17778717", "syn21636326", "syn21636350", "syn21636364")
```

## Running the baseline method

The repository comes with a simple baseline method that demonstrates how to produce predictions in a leave-one-specimen-out setting and the format of these predictions. The following command will train a model on Lung1&2 and apply that model to make predictions on Lung3:

```
Rscript baseline/xgboost.R data/Lung[12].csv.gz data/Lung3.csv.gz
```

The predictions will appear in two-column CSV format as `Lung3-xgboost.csv.gz`. This is the format expected by the scoring script introduced below. We can inspect the first few entries in this file:

```
$ gunzip -c Lung3-xgboost.csv.gz | head
CellID,Pred
1,Stroma
2,Immune
3,Immune
4,Immune
5,Immune
6,Immune
7,Immune
8,Immune
9,Immune
```

## Scoring predictions

To score predictions, we simply provide the two-column CSV file together with the matching "ground truth" file to `score.R`:

```
$ Rscript score.R Lung3-xgboost.csv.gz data/Lung3.csv.gz 
Probability-based AUC values

Probability-based AUC for Immune : 0.6778014 
Probability-based AUC for Stroma : 0.7163537 
Probability-based AUC for  Tumor : 0.7331163 

Confusion Matrix and Statistics

          Reference
Prediction Immune Stroma Tumor
    Immune  53953  19481 13484
    Stroma   3792   6920  1600
    Tumor    3508   1492  6333

...
```

Recall that true labels are assigned based on posterior probability of marker expression. We can also interpret these probabilities as the level of importance for individual predictions. Intuitively, we would like a predictor to be more accurate on cells where we are more confident in our ground truth. Conversely, if we are uncertain that a given cell is Tumor (based on marker expression), it would be unfair to penalize a morphology-based predictor that misclassifies this cell as Stroma or Immune. This is the intuition behind probability-based AUC metrics, which rank all cells according to the correpsonding posterior probability values and compute area under the ROC curve from the matching predictions.

The remaining metrics are more standard and treat predictions and true labels as discrete class calls.
