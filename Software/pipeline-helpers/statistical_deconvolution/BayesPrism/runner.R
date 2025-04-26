suppressWarnings(library(BayesPrism))
suppressWarnings(library(ggplot2))
suppressWarnings(library(cowplot))
args <- commandArgs(trailingOnly = TRUE)
# 1: bulk counts
# 2: single-cell counts
# 3: threads
# 4: use the first N rows (bulk samples)
# 5: file prefix

# Function to print log messages with a timestamp prefix
logging <- function(message) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(paste(timestamp, message, "\n"))
}

# Function to extract first substring from a concatenated string separated by "."
extract_first_substring <- function(string) {
  substrings <- strsplit(string, "\\.|-")[[1]]
  first_substring <- substrings[1]
  return(first_substring)
}

bk.dat = read.csv(args[1], row.names = 1)
if (length(args) >= 4 && args[4] != "none" && args[4] != "all" && !is.null(args[4])){
    param = as.numeric(args[4])
    if (param <= dim(bk.dat)[1]){
        bk.dat = bk.dat[rownames(bk.dat)[1:param],]
        print(paste("Using the first", param, "rows in the bulk dataframe"))
    }else{
        print("Requested rows/samples is larger than actual rows in the bulk dataframe. Parameter ignored.")
    }
}
if (length(args) >= 5){
    suffix = args[5]
}else{
    suffix = ""
}
sc.dat = read.csv(args[2], row.names = 1)

logging("Run BayesPrism")
cell.type.labels <- sapply(rownames(sc.dat), extract_first_substring)
names(cell.type.labels) = NULL

cell.state.labels <- sapply(rownames(sc.dat), extract_first_substring)
names(cell.state.labels) = NULL

# set key to NULL since there are no malignant cells or the malignant cells
# between reference and mixture are from matched samples, in which case
# all cell types will be treated equally.
myPrism <- new.prism(
    reference=sc.dat,
    mixture=bk.dat,
    input.type="count.matrix",
    cell.type.labels = cell.type.labels,
    cell.state.labels = cell.state.labels,
    key=NULL,
    outlier.cut=0.01,
    outlier.fraction=0.1,
)

bp.res <- run.prism(prism=myPrism, n.cores=as.numeric(args[3]))

# extract posterior mean of cell type fraction theta
theta <- get.fraction(bp=bp.res, which.theta="final", state.or.type="type")
# extract posterior mean of cell type-specific gene expression count matrix Z
z = get.exp(bp=bp.res, state.or.type = "type", cell.name=unique(cell.type.labels))
dim_z = dim(z)

logging("Posterior mean of cell type fractions:")
logging(theta)

if (suffix != ""){
    theta_file_name = paste("theta", suffix, "csv", sep=".")
    rdata_file_name = paste("bp.res", suffix, "rdata", sep=".")
    z_file_name = paste("Z", suffix, "csv", sep=".")
    zs_file_pref = paste("Z", suffix, sep=".")
}else{
    theta_file_name = "theta.csv"
    rdata_file_name = "bp.res.rdata"
    z_file_name = "Z.csv"
    zs_file_pref = "Z"
}

write.csv(theta, file = theta_file_name)
save(bp.res, file = rdata_file_name)
if (length(dim_z) == 2){  # dim of z: n_regions, n_cell_types
    write.csv(z, file = z_file_name)
}else{  # dim of z: n_bulk_samples, n_regions, n_cell_types
    for (i in 1:dim_z[1]){
        z_mat = z[i, , ]
        if (i == 1) {
            write.csv(z_mat, file = z_file_name)
        }else{
            write.csv(z_mat, file = paste(zs_file_pref, i, "csv", sep="."))
        }
    }
}
