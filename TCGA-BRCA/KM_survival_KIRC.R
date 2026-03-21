library(TCGAbiolinks)
library(survminer)
library(survival)
library(SummarizedExperiment)
library(tidyverse)
library(DESeq2)

# getting clinical data for TCGA-KIRC cohort -------------------
clinical_kirc <- GDCquery_clinic("TCGA-KIRC")
any(colnames(clinical_kirc) %in% c("vital_status", "days_to_last_follow_up", "days_to_death"))
which(colnames(clinical_kirc) %in% c("vital_status", "days_to_last_follow_up", "days_to_death"))
clinical_kirc[,c(9,39,45)]

# looking at some variables associated with survival 
table(clinical_kirc$vital_status)

# days_to_death, that is the number of days passed from the initial diagnosis to the patient’s death (clearly, this is only relevant for dead patients)
# days_to_last_follow_up that is the number of days passed from the initial diagnosis to the last visit.

# change certain values the way they are encoded
clinical_kirc$deceased <- ifelse(clinical_kirc$vital_status == "Alive", FALSE, TRUE)

# create an "overall survival" variable that is equal to days_to_death
# for dead patients, and to days_to_last_follow_up for patients who
# are still alive
clinical_kirc$overall_survival <- ifelse(clinical_kirc$vital_status == "Alive",
                                         clinical_kirc$days_to_last_follow_up,
                                         clinical_kirc$days_to_death)

# get gene expression data -----------

# build a query to get gene expression data for entire cohort
# Transcriptome Profiling + Gene Expression Quantification:
# High dimensional gene count data
query_kirc_all = GDCquery(
  project = "TCGA-KIRC",
  data.category = "Transcriptome Profiling", # parameter enforced by GDCquery
  experimental.strategy = "RNA-Seq",
  workflow.type = "STAR - Counts",
  data.type = "Gene Expression Quantification",
  sample.type = "Primary Tumor",
  access = "open")

output_kirc <- getResults(query_kirc_all)
output_kirc

# Download all the counting files queried above
# GDCdownload(query_kirc_all)

# get counts
tcga_kirc_data <- GDCprepare(query_kirc_all, summarizedExperiment = TRUE)
# Saves the exact state of the object to your working directory
# NOT NEEDED! Data can be extracted directly from the GDCprepare function
# saveRDS(tcga_kirc_data, file = "tcga_kirc_raw_counts.rds")

# If p: number of genes and n: number of patients, this matrix is p \times n
# For each cell of this table, counts how many times that specific gene appeared
# in the patient's primary tumor genome sequencing
kirc_matrix <- assay(tcga_kirc_data, "unstranded")
kirc_matrix[1:10,1:10]

# extract gene and sample metadata from summarizedExperiment object
# Gene (features) information. p \times m dataset
gene_metadata <- as.data.frame(rowData(tcga_kirc_data))

# Patients information. n \times k dataset
# Each row corresponds to a patient (column) in kirc_matrix
# Contains all the survival times, event indicators, gender, tumor stage, barcode IDs
coldata <- as.data.frame(colData(tcga_kirc_data))

# Event indicator (delta): inside colData - 1 = Dead, 0 = Censored
colData(tcga_kirc_data)$delta <- ifelse(tcga_kirc_data$vital_status == "Dead", 1, 0)
# Survival time (time): inside colData
colData(tcga_kirc_data)$time <- ifelse(
  tcga_kirc_data$vital_status == "Alive",
  tcga_kirc_data$days_to_last_follow_up,
  tcga_kirc_data$days_to_death
)
# Patients whose lifetime is equal to zero may have died on the first day since
# they were diagnosed. To avoid problems, replace zeros by 1
colData(tcga_kirc_data)$time <- ifelse(
  tcga_kirc_data$time == 0,
  1,
  tcga_kirc_data$time
)
# 
# y_time_coxnnet <- read.csv("ytime_coxnnet.csv")
# hist(y_time_coxnnet$X385)
# hist(tcga_kirc_data_study$time)

tcga_kirc_data_study <- tcga_kirc_data
# Filter oout patients who do not have time and censorship information
keep_valid_patients <- which(!is.na(tcga_kirc_data$time) &
                             !is.na(tcga_kirc_data$delta) &
                             (tcga_kirc_data$time > 0))
tcga_kirc_data_study <- tcga_kirc_data[ , keep_valid_patients ]
# Total of 1074 valid patients
dim(tcga_kirc_data_study)

View(as.data.frame(colData(tcga_kirc_data_study))[,c("delta", "time")])

gene_metadata <- as.data.frame(rowData(tcga_kirc_data_study))
coldata <- as.data.frame(colData(tcga_kirc_data_study))
kirc_matrix <- assay(tcga_kirc_data_study, "unstranded")
kirc_matrix[1:10,1:10]

# vst transform counts to be used in survival analysis ---------------

# Setting up countData object
# design is the formula used to tell the object how will we build the design
# matrix for its internal for its internal Negative Binomial GLMs
dds <- DESeqDataSetFromMatrix(countData = kirc_matrix,
                              colData = coldata,
                              design = ~ 1)

# A gene is only biologically relevant to the population if it is meaningfully
# expressed in a reasonable fraction of that population.
# We take out genes that are expressed in less than 5% of the patients overall genes

# Minimal number of patients that can express a gene for it to be considered
min_patients <- round(0.05 * ncol(dds))
min_patients

# We keep only the genes that appear more than 10 times in at least 5% of all
# the patients. The number 10 is called the Limit of Quantification (LOQ)
keep_genes <- rowSums(counts(dds) >= 10) >= min_patients
# The amount of genes considered is drastically reduced with this filter
# We are able to remove most completely unrelated genes from the analysis before
# passing those to a machine learning driven model
sum(keep_genes)

dds <- dds[keep_genes,]
dim(dds)
dds_matrix <- assay(dds)

X_transposed_log <- t( log(1 + dds_matrix) )
time <- dds$time
delta <- dds$delta

final_dataset_log <- data.frame(time = time, delta = delta, X_transposed_log)
write.csv(final_dataset_log, file = "tcga_kirc_count_data_coxnnet.csv", row.names = FALSE)
write.csv(gene_metadata, file = "gene_metadata_coxnnet.csv", row.names = FALSE)


# Perform a Variance Stabilizing Transformation, which standardize count data
# variances according to a Negative Binomial model
vsd <- vst(dds, blind = FALSE)
kirc_matrix_vst <- assay(vsd)
kirc_matrix_vst[1:10,1:10]
dim(kirc_matrix_vst)

X_transposed <- t(kirc_matrix_vst)
time <- vsd$time
delta <- vsd$delta

final_dataset <- data.frame(time = time, delta = delta, X_transposed)
write.csv(final_dataset, file = "tcga_kirc_count_data.csv", row.names = FALSE)
write.csv(gene_metadata, file = "gene_metadata.csv", row.names = FALSE)
