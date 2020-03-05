## Identifies confident "ground truth" cases that were not predicted correctly.
##
## by Artem Sokolov

suppressMessage(library( tidyverse ))

## Load the set of predictions and true labels
cmd <- commandArgs(trailingOnly=TRUE)
if(length(cmd) < 2)
    stop( "Usage: Rscript miss.R <predictions> <true labels>" )

fnPred <- cmd[1]
fnTrue <- cmd[2]

## Load individual files and combine everything into a common data frame
read_csv2 <- partial( read_csv, col_types=cols(CellID = col_integer()) )
Pr <- read_csv2( fnPred )
Tr <- read_csv2( fnTrue )
X <- inner_join(Pr, Tr, by="CellID")

## Identify mis-classified cases that have high confidence in true labels
fnOut <- basename(fnPred) %>% str_split("-") %>%
    pluck(1,1) %>% str_c( "-miss.csv" )
X %>% filter( Pred != Label ) %>%
    filter( Immune > 0.95 | Tumor > 0.95 | Stroma > 0.95 ) %>%
    select( CellID, Pred, Label, Immune, Stroma, Tumor ) %>%
    write_csv( fnOut )

