#!/usr/bin/env Rscript
## Scores a set of {stroma, immune, tumor} predictions against true labels
##
## by Artem Sokolov

suppressMessages(library( tidyverse ))

## Parse arguments: expect two files (one predictions, one true labels)
cmd <- commandArgs(trailingOnly=TRUE)
if(length(cmd) != 2)
    stop( "Usage: Rscript score.R <predictions> <true labels>" )

## Cell types of interest
ct <- c("Immune", "Stroma", "Tumor")

## Load both files and identify the one with predictions
XX <- map( cmd, read_csv, col_types=cols() )
i <- which( map_int(XX, ncol) == 2 )
if( length(i) < 1 ) stop( "Predictions must be in two-column format." )
if( length(i) > 1 ) stop( "True labels file is in the wrong format" )
Xpred <- nth(XX, i) %>% select( CellID, everything() )
Xtrue <- nth(XX, -i) %>% select( CellID, Label, one_of(ct) )

## Give a consistent name to the predictions column and join against labels
R <- rename(Xpred, Pred=2) %>% inner_join(Xtrue, by="CellID")

## An AUC estimate that doesn't require explicit construction of an ROC curve
## Source: Hand & Till (2001)
auc <- function( probs, preds, ref )
{
    stopifnot( length(probs) == length(preds) )
    jp <- which(preds==ref); np <- length(jp)
    jn <- which(preds!=ref); nn <- length(jn)
    s0 <- sum( rank(probs)[jp] )
    (s0 - np*(np+1) / 2) / np / nn
}

## Compute probability based AUC (pauc) values for each cell type of interest
cat("Probability-based AUC values\n\n")
set_names(ct) %>% map_dbl(~auc(R[[.x]], R$Pred, .x)) %>%
    iwalk(~cat("Probability-based AUC for", str_pad(.y,6), ":", .x, "\n"))

## Display other statistics
cat("\n")
with( R, caret::confusionMatrix(factor(Pred, ct),
                                factor(Label, ct),
                                mode="everything") )
