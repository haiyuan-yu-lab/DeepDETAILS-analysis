args <- commandArgs(trailingOnly = TRUE)
# Code between # <~~~~~~~ and # ~~~~~~~> comes from the DSA release version 1.0
# <~~~~~~~
# ---------------------------------------------------------------------------------------------------------------------------------------------------
#	Function: DSA
#	Desription: This is the main interface function to call the DSA
#	Required arguments:
#		mix					matrix of mixture signals (genes in row, cell type in column).
#		cell.gene			the cell-gene key table: mapping between cell type and cell specific genes. This table needs to be only two columns: first column with gene symbols, second column the cell type.
# 	Optional argumanets:
#		weight				weight matrix (samples in row, cell types in column).
#							Default to 'NULL', where no weight matrix is provided by user, and the weight will be estimated by the function.
#		method				methods used in estimating weight and true signals for each cell type.  Default to 'LM' for linear regression.
#							Other methods included 'LG' (logistic regression),'QP_LM' or 'QP_LG' (quadradic programming with constraint on the estimated parameter on linear/logistic regression)
#		out.cell.file		file name to store the deconvoluted signals.  default to 'NULL' where no file will be created.
#		out.weight.file		file name to store the estimated weights.  default to 'NULL' where no file will be created.
#		log2				flag indicating if the input mixture signals are in log2 scale.  Default to 'TRUE'.
#		l, u		values for the lower- (l) and upper- (u) bound used in setting the vector for values of b0 (bvec for solve.QP). Defaults to 0 and 2^34 respectively.
#		meq			default to zero (used to set meq for solve.QP)
#	Returned values:
#		est.weight			a list of estimated weight ('estimated_weight') and the model's mean square error ('mse'). estimated_weight is a matrix of cell types in row and samples in columns. mse defaults to 'NULL' if method is not 'LM'.
#		deconv				a matrix of deconvoluted signals: genes in row, cell types in columns.
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------
DSA <- function(mix, cell.gene, weight = NULL, method = "LM", out.cell.file = NULL, out.weight.file = NULL, log2 = TRUE, l = 0, u = 2^34, meq = 0)
{
	#mix_file <- as.matrix(read.delim(file = mix.signal.file, header = TRUE, sep = "\t", quote="\"", dec=".", fill = TRUE, row.names=1))
	if(log2 == FALSE){
		mix <- log2(mix)
	}

	#cell_gene_table <- as.matrix(read.table(file = cell.gene.table, header = FALSE, sep = "\t", quote="\"", dec=".", fill = TRUE))
	if(ncol(cell.gene) != 2){
		print("Error: The cell-gene key table should be a table with only two columns.")
		stop()
	}

	unique_cell_type <- unique(cell.gene[,2])
	gene_list <- list()
	for( i in 1 : length(unique_cell_type)){
		gene_list[[i]] <- cell.gene[cell.gene[,2] == unique_cell_type[i],1]
	}
	names(gene_list) <- unique_cell_type

	estimated_weight <- list()
	if(is.null(weight)){
		estimated_weight <- EstimateWeight(2^mix, gene_list, method="LM", l, u)
	}
	else {	## Read the weight matrix from file
		#estimate_weight <- (as.matrix(read.delim(file = weight.file, header = TRUE, sep = "\t", fill = TRUE)))
		estimate_weight <- as.matrix(weight)
		#estimate_weight <- estimate_weight/colSums(estimate_weight)
		estimate_weight <- estimate_weight/rowSums(estimate_weight)

		estimated_weight$weight <- t(estimate_weight)
		estimated_weight$mse <- NULL
	}

	## Get the deconvoluted signals
	ten_cell_decon_QP_LM <- Deconvolution(2^mix, t(estimated_weight$weight), method, l, u)
	rownames(ten_cell_decon_QP_LM) <- rownames(mix)
	colnames(ten_cell_decon_QP_LM) <- unique_cell_type

	if (!is.null(out.cell.file)){
		write.table(log2(ten_cell_decon_QP_LM), file = out.cell.file, append = FALSE, quote = FALSE, sep = "\t", eol = "\n", na = "NA", dec = ".", row.names = TRUE, col.names = TRUE)
	}
	if (!is.null(out.weight.file)){
		write.table(estimated_weight$weight, file = out.weight.file, quote = FALSE, sep = "\t", eol = "\n", na = "NA", dec = ".", row.names = TRUE, col.names = TRUE)
	}

	return (list(est.weight=estimated_weight$weight, deconv=log2(ten_cell_decon_QP_LM)))
}

# ---------------------------------------------------------------------------------------------------------------------------------------------------
#	Function: Deconvolution
#	Desription: Function to estimate the deconvoluted signals - deconvolve the mixture signals to cell-type specific signals for each gene.
#	Required arguments:
#		data		data matrix of the mixture signals, with genes in row, cell type in column, in anti-log scale.
#		weight		weight matrix, with cell type in row, tissue types in column.
# 	Optional argumanets:
#		method		methods used in estimating true signals for each tissue type.  Default to 'LM' for linear regression.
#					Other methods included 'LG' (logistic regression),'QP_LM' or 'QP_LG' (quadradic programming with constraint on the estimated parameter on linear/logistic regression)
#					For methods 'LG' and 'QP_LG', input data is transformed into log-scaled and returned values are anti-logged.
#		l, u		values for the lower- (l) and upper- (u) bound used in setting the vector for values of b0 (bvec for solve.QP). Defaults to 0 and 2^34 respectively.
#	Returned values:
#		paraM		a matrix of deconvoluted signals with genes in row, tissue types in columns
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------
Deconvolution <- function(data, weight, method = "LM", l = 0, u = 2^34)
{
	print("Deconvolution ...")
	i <- 1
	data <- as.matrix(data)
	weight <- as.matrix(weight)
	paraM <- matrix(0, nrow(data), ncol(weight))

	if( method == "LM"){
		for(i in 1 : nrow(data)){
			y <- data[i,]
			lmOb <- lm(y ~ -1 + weight)
			paraM[i,] <- lmOb$coefficients
		}
	}

	else if (method == "QP_LM"){
		paraM <- GSM_QP(data, weight, l = min(data), u = max(data) , meq =0)
	}

	else if (method == "LG"){
		for(i in 1 : nrow(data)){
			y <- log2(data[i,])
			lmOb <- lm(y ~ -1 + weight)
			paraM[i,] <- 2^lmOb$coefficients
		}
	}

	else if ( method == "QP_LG"){
		data <- log2(data)
		paraM <- GSM_QP(data, weight, l = min(data), u = max(data) , meq =0)
		paraM <- 2^paraM
	}

	else{
		print("Error: Deconvolution method parameter is not supported. Methods supported: 'LM', 'QP_LM', 'LG', 'QP_LG'.")
		stop()
	}

	return (paraM)
}

## this function I will select the number of specific gene for each cell type, by the proportion
## parameter Qp_linear is a logical variable with false linear model, true as QP model
## this estimate is better
# ---------------------------------------------------------------------------------------------------------------------------------------------------
#	Function: EstimateWeight
#	Desription: Function to estimate the weight matrix.  Based on the set of marker genes for each cell type, this function estimate the cell-specific proportions (weight) for each sample.
#	Required arguments:
#		mix_ob			data matrix of the mixture signals, with genes in row, cell type in column, in anti-log scale.
#		gene_list		list of the length in the number of tissue types. Each list element contains gene symbols representing the tissue type
# 	Optional argumanets:
#		method			methods used in estimating true signals for each tissue type.  Default to 'LM' for linear regression.
#						Other methods included 'QP_LM' (quadradic programming with constraint on the estimated parameter on linear regression)
#		l, u		values for the lower- (l) and upper- (u) bound used in setting the vector for values of b0 (bvec for solve.QP). Defaults to 0 and 2^34 respectively.
#	Returned values:
#		weight			matrix of estimated weight, with cell type in row, tissue types in column.
#		mse				means square error of the fitted linear model.  mse is 'NULL' if method is 'QP_LM'
#	Remarks:this function I will select the number of specific gene for each cell type, by the proportion (??)
## parameter Qp_linear is a logical variable with false linear model, true as QP model
# ---------------------------------------------------------------------------------------------------------------------------------------------------
EstimateWeight <- function(mix_ob, gene_list, method = "LM", l = 0, u = 2^34)
{
	print("Estimate Weight...")
	select_mix_ob <- matrix()
	for ( i in 1 : length(gene_list)){
		if(i == 1) {
			if(length(gene_list[[i]]) == 1){
				select_mix_ob <- as.matrix(mix_ob[gene_list[[i]], ])
			}
			else{
				select_mix_ob <-  as.matrix(colMeans( mix_ob[gene_list[[i]], ]))
			}
		}
		else {
			if(length(gene_list[[i]]) == 1){
				select_mix_ob <- cbind(select_mix_ob, as.matrix( mix_ob[gene_list[[i]], ]))
			}
			else {
				select_mix_ob <- cbind(select_mix_ob, as.matrix(colMeans( mix_ob[gene_list[[i]], ])))
			}
		}
	}

	y <- rep(1, times = nrow(select_mix_ob))

	b_par <- numeric()
	mse <- numeric()

	## we will use linear fit or QP method to get the estimator of parameter
	if(method != "LM" && method != "QP_LM"){
		print("Error: Weight estimation method not supported. Methods supported are: 'LM' or 'QP'.")
		stop()
	}
	if(method == "LM") {
		lmob <- lm( y ~  -1 + (select_mix_ob))
		mse <- sum(lmob$residuals^2)
		b_par <- coef(lmob)
	}
	else {
		y <- rep(1, times = nrow(select_mix_ob))
		y <- t(as.matrix(y))

		Qp <- GSM_QP(y, (select_mix_ob), l, u)
		b_par <- Qp
		mse <- NULL
	}

	len <- as.integer(length(gene_list))

	par_matrix <- diag(c(b_par), len, len)
	estimate_weight <- par_matrix %*% t((select_mix_ob ))
	rownames(estimate_weight) <- names(gene_list)

	return(list(weight=estimate_weight, mse=mse))
}
# this function is about the quadratic programming
# ---------------------------------------------------------------------------------------------------------------------------------------------------
#	Function: GSM_QP
#	Desription: Estimate the true signals for each tissue type with quadratic programming.
#	Required arguments:
#		ob				data matrix of the mixture signals, with genes in row, cell type in column.
#		weight		weight matrix, with cell type in row, tissue types in column.
#		l, u		values for the lower- (l) and upper- (u) bound used in setting the vector for values of b0 (bvec for solve.QP)
# 	Optional argumanets:
#		meq			default to zero (used to set meq for solve.QP)
#	Returned values:
#		sol			estimated true signals for each gene (in row) in each cell type (in column)
#	Remarks:	this functions depends on the solve.QP function from quadprog package.
#	See also:	solve.QP
# ---------------------------------------------------------------------------------------------------------------------------------------------------
GSM_QP <- function(ob, weight, l = 0, u = 2^34, meq = 0)
{
	require("quadprog")

	sol <- c()
	ob <- as.matrix(ob)
	weight <- as.matrix(weight)

	for (id in 1:nrow(ob)){
		A <- weight
		b <- ob[id, ]
		Dmat <- t(A) %*% A
		dvec <- b %*% (A)
		numC <- ncol(weight)
		Amat <- diag(rep(1, numC))
		Amat <- rbind(Amat, diag(rep(-1, numC)))
		Amat <- t(Amat)
		bvec <- c(rep(l, numC), rep(-u, numC))

		if(meq > 0){
			Amat <- cbind(rep(1,numC),Amat)
			bvec <- c(1,bvec)
		}

		sol <- rbind(sol, solve.QP(Dmat, dvec, Amat, bvec=bvec, meq=meq)$solution)
	}
	return (sol)
}
# ~~~~~~~>
# Function to find the column index with the maximum value for each row
find_max_column <- function(row) {
  return(which.max(row))
}

mix = read.csv(args[1], row.names = 1)
sig.dat = read.csv(args[2], sep = "\t")
row.mapping = read.csv(args[3], row.names = 1)

processed_sig = merge(row.mapping, sig.dat, by.x = "mapped", by.y = "NAME", all = F)
processed_sig <- processed_sig[, -which(names(processed_sig) == "mapped")]
# Set the "ID" column as row names
rownames(processed_sig) <- processed_sig$original
# Drop the "ID" column (optional, if you want to remove the column after setting it as row names)
processed_sig <- processed_sig[, -which(names(processed_sig) == "original")]
# Figure out in which cell type the signature has the most abundant expression
max_columns <- apply(processed_sig, 1, find_max_column)
# Prepare the "list" of signatures in DSA's format
cell.gene <- cbind(rownames(processed_sig), colnames(processed_sig)[max_columns])
colnames(cell.gene) <- c("V1", "V2")

# shape of mix: n_genes x n_mixtures
#   rownames should be gene/region name/ID
# shape of cell.gene: n_sig x 2
#   first column stores the name of the signature.
#   second column stores which cell type the signature serves for.
# set log2 as FALSE, since our read counts are not log2-transformed before calling the coming function
result = DSA(mix, cell.gene, weight = NULL, method = "LM", out.cell.file = NULL, out.weight.file = NULL, log2 = FALSE, l = 0, u = 2^34, meq = 0)

# fractions should be a matrix with shape n_types x n_mixtures
fractions = result$est.weight
write.csv(fractions, "fractions.csv", quote = FALSE)

# GEPs should be a matrix with shape n_gene x n_types
geps = result$deconv
# export unscaled group estimation
write.csv(geps, "geps.csv", quote = FALSE)

# export scaled counts table for each sample
for (i in 1:ncol(fractions)) {
  # Multiply the current column of fractions with geps
  result <- fractions[,i] * geps

  write.csv(geps, paste("geps", "scaled", i, "csv", sep="."), quote = FALSE)
}