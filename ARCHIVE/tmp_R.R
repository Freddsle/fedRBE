library(tidyverse)

pg_matrix <- read.csv("/home/yuliya/repos/cosybio/removeBatch/test_data/all_pg_matrix.csv", row.names = 1)
dim(pg_matrix)
head(pg_matrix)

a_prots <- rownames(pg_matrix[,batch_info[batch_info$lab == 'lab_A',]$file] %>% filter(rowSums(is.na(.)) != ncol(.)))
c_prots <- rownames(pg_matrix[,batch_info[batch_info$lab == 'lab_C',]$file] %>% filter(rowSums(is.na(.)) != ncol(.)))
d_prots <- rownames(pg_matrix[,paste0('X', batch_info[batch_info$lab == 'lab_D',]$file)] %>% filter(rowSums(is.na(.)) != ncol(.)))
e_prots <- rownames(pg_matrix[,batch_info[batch_info$lab == 'lab_E',]$file] %>% filter(rowSums(is.na(.)) != ncol(.)))

shared_prots <- intersect(a_prots, c_prots)
shared_prots <- intersect(shared_prots, d_prots)
shared_prots <- intersect(shared_prots, e_prots) %>% sort()

pg_matrix <- pg_matrix[shared_prots, ]

batch_info <- read.csv("/home/yuliya/repos/cosybio/removeBatch/test_data/batch_info_all.csv") %>%
  mutate(lab = factor(lab), condition = factor(condition))
rownames(batch_info)<- batch_info$file
dim(batch_info)
head(batch_info)

library(limma)
pg_matrix_cured <- removeBatchEffect(pg_matrix, batch_info[['lab']]) %>% as.data.frame()


################################################################################################################

batch <- as.factor(batch_info[['lab']])
contrasts(batch) <- contr.sum(levels(batch))
batch <- model.matrix(~batch)[,-1,drop=FALSE]
batch

x <- as.matrix(pg_matrix)
design=matrix(1,ncol(x),1)
batch.X <-  cbind(design, batch)
fit <- lmFit(x, batch.X)

b <- fit$coefficients
beta <- fit$coefficients[, -(1:ncol(design)), drop=FALSE]

beta[is.na(beta)] <- 0
pg_matrix_cured_2 <- x - beta %*% t(batch) %>% as.data.frame()

################################################################################################################
removeBatchEffect <- function(x,batch=NULL,batch2=NULL,covariates=NULL,design=matrix(1,ncol(x),1),group=NULL,...)
{
  #	Covariates to remove (batch effects)
  if(is.null(batch) && is.null(batch2) && is.null(covariates)) return(as.matrix(x))
  if(!is.null(batch)) {
    batch <- as.factor(batch)
    contrasts(batch) <- contr.sum(levels(batch))
    batch <- model.matrix(~batch)[,-1,drop=FALSE]
  }
  if(!is.null(batch2)) {
    batch2 <- as.factor(batch2)
    contrasts(batch2) <- contr.sum(levels(batch2))
    batch2 <- model.matrix(~batch2)[,-1,drop=FALSE]
  }
  if(!is.null(covariates)) covariates <- as.matrix(covariates)
  X.batch <- cbind(batch,batch2,covariates)
  
  #	Covariates to keep (experimental conditions)
  if(!is.null(group)) {
    group <- as.factor(group)
    design <- model.matrix(~group)
  }
  
  #	Fit combined linear model
  x <- as.matrix(x)
  fit <- lmFit(x,cbind(design,X.batch),...)
  
  #	Subtract batch effects adjusted for experimental conditions
  beta <- fit$coefficients[,-(1:ncol(design)),drop=FALSE]
  beta[is.na(beta)] <- 0
  x - beta %*% t(X.batch)
}