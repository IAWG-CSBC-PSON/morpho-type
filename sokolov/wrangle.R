## Wrangles feature files and derives labels
##
## by Artem Sokolov
##
## GitHub dependencies:
##  - Sage-Bionetworks/synapser
##  - ArtemSokolov/synExtra
##  - labsyspharm/naivestates

library( tidyverse )
library( here )

synapser::synLogin()
syn <- synExtra::synDownloader(here("data","raw"))

## Feature sets of interest (f*)
fExpr <- c("KERATIN", "ASMA", "CD45")
fGeom <- c("Area", "Eccentricity", "Solidity", "Extent", "EulerNumber", 
           "Perimeter", "MajorAxisLength", "MinorAxisLength", "Orientation")
fNeig <- str_c("Neighbor_", 1:5)
fPos  <- c("X_position", "Y_position")

## Identify, download, and load all source feature files
Xraw <- c(Lung1 = "syn19035296",
          Lung2 = "syn19035616",
          Lung3 = "syn19035696") %>% map(syn) %>%
    map( read_csv, col_types=cols(CellID=col_integer()) )

## Define quantiles (QQ) for outlier removal
QQ <- list(
    Lung1 = list(KERATIN=c(0.001,0.002), ASMA=c(0.01,0),  CD45=c(0.002,0.002)),
    Lung2 = list(KERATIN=c(0.001,0),     ASMA=c(0.005,0), CD45=c(0.003,0.001)),
    Lung3 = list(KERATIN=c(0.001,0),     ASMA=c(0.015,0), CD45=c(0.001,0.001)))

## Fits a mixture of two Gaussians to each channel in X using quantiles in Q
f <- function(X, Q) map(fExpr, ~naivestates::GMMfit(X, CellID, .x, qq=Q[[.x]]))
stopifnot( identical(names(QQ), names(Xraw)) )
Fits <- map2(Xraw, QQ, f) %>% map(bind_rows)

## Marker -> Cell type (mct) association
mct <- c("Immune" = "CD45", "Stroma" = "ASMA", "Tumor" = "KERATIN")
ct <- names(mct)                   ## Cell types: Immune, Stroma, Tumor
ect <- expr( list(!!!syms(ct)) )   ## Expression: list(Immune, Stroma, Tumor)    

## Given a fit frame F, extracts posterior probabilities and makes class calls
fcc <- function(F) {
    naivestates::GMMreshape(F) %>% rename(!!!mct) %>%
        mutate(Sum    = pmap_dbl(!!ect, sum),
               Argmax = pmap_int(!!ect, lift_vd(which.max)),
               Label  =  map_chr(Argmax, ~ct[.x])) %>%
        mutate_at( ct, ~.x/Sum ) %>%
        select(CellID, Label, one_of(ct))
}

## Derive class call labels for each dataset and join with input features
Labels <- map( Fits, fcc )

## Combine input features and labels into a common dataframe
stopifnot( identical(names(Labels), names(Xraw)) )
Data <-map( Xraw, select, CellID, one_of(c(fGeom, fNeig, fPos)) ) %>%
    map2( Labels, ., inner_join, by="CellID" )

## Save to files (assign to tmp to suppress stdout)
tmp <- imap( Data, ~write_csv(.x, here("data",str_c(.y,".csv.gz"))) )
