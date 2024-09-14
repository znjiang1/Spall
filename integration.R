library(Seurat)

inforDir <- './data/PDAC-A/Infor_Data'
sc.count1 <- t(read.csv(paste0(inforDir,"/ST_count/ST_count_1.csv"), header=T, row.names = 1,check.names=FALSE))
sc.count2 <- t(read.csv(paste0(inforDir,"/ST_count/ST_count_2.csv"), header=T, row.names = 1,check.names=FALSE))
st_counts <- list(sc.count1,sc.count2)

c1<- Seurat::CreateSeuratObject(counts = st_counts[[1]])
c2<- Seurat::CreateSeuratObject(counts = st_counts[[2]])
c1<-NormalizeData(c1)
c2<-NormalizeData(c2)
pbmc.obj <- FindIntegrationAnchors(object.list = list(c1,c2))
pbmc.obj <- IntegrateData(anchorset = pbmc.obj)
# integrated.mat <- GetAssayData(object = pbmc.obj, slot = "data", assay = "integrated")
integrated.mat <- as.matrix(GetAssayData(object = pbmc.obj, slot = "data", assay = "integrated"))
integrated.mat.t <- t(integrated.mat)
write.csv(integrated.mat.t,file=paste0(inforDir,'/integrated.csv'),quote=F)