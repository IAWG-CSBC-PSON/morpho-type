# morpho-type
Prediction of cell type from morphology features and neighborhood information

## Installation

```
git clone https://github.com/IAWG-CSBC-PSON/morpho-type.git
```
to get access to the data and scripts in this repository.

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
Raw images from which all features were derived can be downloaded from Synapse in .ome.tif format:

  * Lung1 -> [syn17774887](https://www.synapse.org/#!Synapse:syn17774887)
  * Lung2 -> [syn17776482](https://www.synapse.org/#!Synapse:syn17776482)
  * Lung3 -> [syn17778717](https://www.synapse.org/#!Synapse:syn17778717)
  
