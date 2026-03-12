library(TCGAbiolinks)
library(survminer)
library(survival)
library(SummarizedExperiment)
library(tidyverse)
library(DESeq2)

# getting clinical data for TCGA-BRCA cohort -------------------
clinical_brca <- GDCquery_clinic("TCGA-BRCA")
any(colnames(clinical_brca) %in% c("vital_status", "days_to_last_follow_up", "days_to_death"))
which(colnames(clinical_brca) %in% c("vital_status", "days_to_last_follow_up", "days_to_death"))
clinical_brca[,c(9,39,45)]

# looking at some variables associated with survival 
table(clinical_brca$vital_status)

# days_to_death, that is the number of days passed from the initial diagnosis to the patient’s death (clearly, this is only relevant for dead patients)
# days_to_last_follow_up that is the number of days passed from the initial diagnosis to the last visit.

# change certain values the way they are encoded
clinical_brca$deceased <- ifelse(clinical_brca$vital_status == "Alive", FALSE, TRUE)

# create an "overall survival" variable that is equal to days_to_death
# for dead patients, and to days_to_last_follow_up for patients who
# are still alive
clinical_brca$overall_survival <- ifelse(clinical_brca$vital_status == "Alive",
                                         clinical_brca$days_to_last_follow_up,
                                         clinical_brca$days_to_death)

# get gene expression data -----------

# build a query to get gene expression data for entire cohort
# Transcriptome Profiling + Gene Expression Quantification:
# High dimensional gene count data
query_brca_all = GDCquery(
  project = "TCGA-BRCA",
  data.category = "Transcriptome Profiling", # parameter enforced by GDCquery
  experimental.strategy = "RNA-Seq",
  workflow.type = "STAR - Counts",
  data.type = "Gene Expression Quantification",
  sample.type = "Primary Tumor",
  access = "open")

output_brca <- getResults(query_brca_all)
output_brca

# Download all the counting files queried above
# GDCdownload(query_brca_all)

# get counts
tcga_brca_data <- GDCprepare(query_brca_all, summarizedExperiment = TRUE)
# Saves the exact state of the object to your working directory
# NOT NEEDED! Data can be extracted directly from the GDCprepare function
# saveRDS(tcga_brca_data, file = "tcga_brca_raw_counts.rds")

# If p: number of genes and n: number of patients, this matrix is p \times n
# For each cell of this table, counts how many times that specific gene appeared
# in the patient's primary tumor genome sequencing
brca_matrix <- assay(tcga_brca_data, "unstranded")
brca_matrix[1:10,1:10]

# extract gene and sample metadata from summarizedExperiment object
# Gene (features) information. p \times m dataset
gene_metadata <- as.data.frame(rowData(tcga_brca_data))

# Patients information. n \times k dataset
# Each row corresponds to a patient (column) in brca_matrix
# Contains all the survival times, event indicators, gender, tumor stage, barcode IDs
coldata <- as.data.frame(colData(tcga_brca_data))

# The power of this structure: If I remove a patient from colData,
# the SummarizedObject automatically deletes their corresponding column in the
# assay matrix or if you filter junk genes from rowData, it deletes their
# corresponding rows from the matrix
# As an example, let's consider taking only the male patients data from tcga_brca_data
# which are very few, since we are treating breast cancer cases.
# Also, to showcase we can filter out genes too, we consider only those with source = "HAVANA"
dim(tcga_brca_data)
keep_havana_genes <- which(rowData(tcga_brca_data)$source == "HAVANA")
keep_male_patients <- which( tcga_brca_data$gender == "male" )
# tcga_brca_data$... supports directly the colData columns
# keep_female_patients2 <- which( colData(tcga_brca_data)$gender == "female" )
tcga_brca_data_filtered <- tcga_brca_data[ keep_havana_genes , keep_male_patients ]
as.data.frame(colData(tcga_brca_data_filtered))$gender
as.data.frame(rowData(tcga_brca_data_filtered))$source
View( as.data.frame(colData(tcga_brca_data_filtered)) )


# Event indicator (delta): inside colData - 1 = Dead, 0 = Censored
colData(tcga_brca_data)$delta <- ifelse(tcga_brca_data$vital_status == "Dead", 1, 0)
# Survival time (time): inside colData
colData(tcga_brca_data)$time <- ifelse(
  tcga_brca_data$vital_status == "Alive",
  tcga_brca_data$days_to_last_follow_up,
  tcga_brca_data$days_to_death
)
# Patients whose lifetime is equal to zero may have died on the first day since
# they were diagnosed. To avoid problems, replace zeros by 1
colData(tcga_brca_data)$time <- ifelse(
  tcga_brca_data$time == 0,
  1,
  tcga_brca_data$time
)

tcga_brca_data_study <- tcga_brca_data
# Filter oout patients who do not have time and censorship information
keep_valid_patients <- which(!is.na(tcga_brca_data$time) &
                             !is.na(tcga_brca_data$delta) &
                             (tcga_brca_data$gender == "female") &
                             (tcga_brca_data$time > 0))
tcga_brca_data_study <- tcga_brca_data[ , keep_valid_patients ]
# Total of 1074 valid patients
dim(tcga_brca_data_study)

View(as.data.frame(colData(tcga_brca_data_study))[,c("delta", "time")])

gene_metadata <- as.data.frame(rowData(tcga_brca_data_study))
coldata <- as.data.frame(colData(tcga_brca_data_study))
brca_matrix <- assay(tcga_brca_data_study, "unstranded")
brca_matrix[1:10,1:10]

# vst transform counts to be used in survival analysis ---------------

# Setting up countData object
# design is the formula used to tell the object how will we build the design
# matrix for its internal for its internal Negative Binomial GLMs
dds <- DESeqDataSetFromMatrix(countData = brca_matrix,
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

# Perform a Variance Stabilizing Transformation, which standardize count data
# variances according to a Negative Binomial model
vsd <- vst(dds, blind = FALSE)
brca_matrix_vst <- assay(vsd)
brca_matrix_vst[1:10,1:10]
dim(brca_matrix_vst)

X_transposed <- t(brca_matrix_vst)
time <- vsd$time
delta <- vsd$delta

final_dataset <- data.frame(time = time, delta = delta, X_transposed)
write.csv(final_dataset, file = "tcga_brca_count_data.csv", row.names = FALSE)
