#!/usr/bin/env Rscript
## Trains a baseline model using the provided list of files
##
## by Artem Sokolov

suppressMessages(library(tidyverse))

## Parse arguments: train on the first n-1 files, test on the last
cmd <- commandArgs(trailingOnly=TRUE)
if(length(cmd) < 2)
    stop( "xgboost <files to train and test> <proportion in training sample>." )

## Identify train and test files
prop <- as.numeric(last(cmd))
fnTrain <- setdiff(cmd, prop)
cat("Training on", str_flatten(fnTrain, ", "), "\n")
cat("Training with", prop, " of the data\n")

## Load all input files
Tr = do.call(rbind, lapply(fnTrain, function(file){ out=read_csv(file, col_types=cols() ); out$file=file; out } ) )
# Tr <- map(fnTrain, read_csv, col_types=cols()) %>% bind_rows
Trrows = sample(nrow(Tr), round(prop * nrow(Tr)), replace=FALSE)
Te = Tr[-Trrows,]
Tr = Tr[Trrows,]


## Identify input features
fInput <- c("Area", "Eccentricity", "Solidity", "Extent", "EulerNumber", 
            "Perimeter", "MajorAxisLength", "MinorAxisLength", "Orientation")

## Mapping (m) between class labels (l) and class indices (i)
mli <- c( Stroma = 0L, Immune = 1L, Tumor = 2L )

## Train a model
X <- select(Tr, one_of(fInput)) %>% as.matrix
y <- mli[Tr$Label]
xgbp <- list( objective="multi:softmax", num_class=3 )
mdl <- xgboost::xgboost( X, y, nrounds=100, params=xgbp, print_every_n=10 )

## Apply the model to score test data
Xte <- select(Te, one_of(fInput)) %>% as.matrix
stopifnot( identical(colnames(X), colnames(Xte)) )
ypred <- mli[ match(predict(mdl, Xte), mli) ] %>% names

## Combine with CellIDs and write out to file
Pred <- select(Te, CellID, file) %>% mutate(Pred = ypred)
fnOut <- prop %>% str_split(".csv") %>%
    pluck(1,1) %>% str_c( "-xgboost.csv.gz" )
cat("Saving predictions to", fnOut, "\n")
write_csv( Pred, fnOut )

## gbm alternative
## Xy <- select( Xraw, Label, one_of(fInput) )
## mdl2 <- gbm( Label ~ ., data=Xy, distribution="multinomial", verbose=TRUE )

## Ypred <- predict( mdl, select(Xy, -Label), 100 )[,,1]
## ypred <- colnames(Ypred)[apply( Ypred, 1, which.max )]

## caret::confusionMatrix( factor(ypred, names(mli)),
##                        factor(Xy$Label, names(mli)),
##                        mode="everything" )
