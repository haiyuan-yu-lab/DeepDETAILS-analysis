FROM ubuntu:22.04
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Prevent Docker from outputting to stdout
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgit2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libssl-dev \
    wget \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libgsl-dev \
    libcairo2-dev \
    libxt-dev \
    r-base \
    r-base-dev

# Install devtools
RUN R -e "install.packages('devtools'); if (!library(devtools, logical.return=T)) quit(status=10)"
# Install snowfall
RUN R -e "install.packages('snowfall'); if (!library(snowfall, logical.return=T)) quit(status=10)"
# Install BiocManager
RUN R -e "install.packages('BiocManager'); if (!library(BiocManager, logical.return=T)) quit(status=10)"
RUN R -e "BiocManager::install('Biobase'); if (!library(BiocManager, logical.return=T)) quit(status=10)"
# Install NMF
RUN R -e "install.packages('NMF'); if (!library(NMF, logical.return=T)) quit(status=10)"
# Install gplots
RUN R -e "install.packages('gplots'); if (!library(gplots, logical.return=T)) quit(status=10)"
# Install scran
RUN R -e "BiocManager::install('scran'); if (!library(scran, logical.return=T)) quit(status=10)"
# Install BiocParallel
RUN R -e "BiocManager::install('BiocParallel'); if (!library(BiocParallel, logical.return=T)) quit(status=10)"
# Install BayesPrism
RUN R -e "devtools::install_github('Danko-Lab/BayesPrism/BayesPrism'); if (!library(BayesPrism, logical.return=T)) quit(status=10)"
# Install cowplot
RUN R -e "install.packages('cowplot'); if (!library(cowplot, logical.return=T)) quit(status=10)"
