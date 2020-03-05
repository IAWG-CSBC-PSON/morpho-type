## Identifies packages that need to be installed and installs them
##
## by Artem Sokolov

## Currently installed packages
ipkgs <- rownames(installed.packages())

## CRAN
cran <- c("tidyverse", "devtools", "here", "xgboost", "caret")
install.packages( setdiff(cran, ipkgs) )

## GitHub dependencies
## Synapser
if( !("synapser" %in% ipkgs) )
    install.packages("synapser", repos=c("http://ran.synapse.org", "http://cran.fhcrc.org"))    

## SynExtra
if( !("synExtra" %in% ipkgs) )
    devtools::install_github( "ArtemSokolov/synExtra" )

## naivestates
if( !("naivestates" %in% ipkgs) )
    devtools::install_github( "labsyspharm/naivestates" )
