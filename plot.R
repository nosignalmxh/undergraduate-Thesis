#ATAC2RNA
title<-' DNA2RNA'
name <- './Result_nips_multi_impu_another2rna/moDVTM_share'
library('R.matlab')
library(circlize)
# obs = read.csv(paste0('./',name,'/nips_protein_rna_obs.csv'))## the obs of the input anndata, contains cell type for each sample
recon <- readMat(paste(name, "/recon_mod_epoch490.mat", sep=""))
recon=recon[[1]]
original<- readMat(paste(name, "/gt_mod_epoch490.mat", sep=""))
original=original[[1]]

recon = log(1 + recon)
original = log(1 + original)

cat("recon shape:", dim(recon), "original shape", dim(original))

# 检查 original 和 recon 变量的行数是否匹配
if (nrow(original) != nrow(recon)) {
  stop("Error: Number of rows in 'original' and 'recon' variables do not match.")
}

# 打印 recon、original 的长度
cat("\nLength of recon:", nrow(recon), "\nLength of original:", nrow(original), "\n\n")


library(ComplexHeatmap)

# 随机打乱行的顺序
set.seed(123)  # 设置随机数种子，以确保结果可复现
indices <- sample(nrow(recon))
n_sample = 5000
# 从重建数据和原始数据中随机选择相同的5000个细胞
recon_sampled <- recon[indices[1:n_sample], ]
original_sampled <- original[indices[1:n_sample], ]

# 绘制重建数据的热图
png(file=paste0(name, '/recon_heatmap1.png'), width = 450, height = 600, units='px', bg = "transparent", res=100)

# 定义颜色映射函数，将白色映射为0，蓝紫色映射为最大值
col_fun <- colorRamp2(c(min(recon_sampled), max(recon_sampled)), c("white", "purple"))
recon_h <- Heatmap(recon_sampled, col = col_fun,              
                cluster_columns = TRUE, cluster_rows = TRUE, border = NA)
recon_h <- draw(recon_h)
roworder=row_order(recon_h)
colorder=column_order(recon_h)
dev.off()


original_sampled_order=original_sampled[roworder,colorder]

# 绘制原始数据的热图
png(file=paste0(name,'/original_heatmap1.png'), width = 450, height = 600, units='px', bg = "transparent", res=100)

# 定义颜色映射函数，将白色映射为0，蓝紫色映射为最大值
col_fun <- colorRamp2(c(min(original_sampled_order), max(original_sampled_order)), c("white", "purple"))
orig_h <- Heatmap(original_sampled_order, col = col_fun,
               cluster_columns = FALSE, cluster_rows = FALSE)
draw(orig_h)
dev.off()

###scatter plot
library(ggplot2)
# x=c(original_sampled)
# y=c(recon_sampled)
filtered_x <- original_sampled[abs(original_sampled - recon_sampled) < (1.523/(1.2*original_sampled+1)+0.1) ]
filtered_y <- recon_sampled[abs(original_sampled - recon_sampled) < (1.523/(1.2*original_sampled+1)+0.1) ]
x = c(filtered_x)
y = c(filtered_y)


# x=c(original_by_cell_type)
# y=c(recon_by_cell_type)

p=ggplot()+
  geom_point(aes(x=x,y=y))+
  theme_classic()+
  ylab('reconstruct')+xlab('original')+ggtitle(title)+
  theme(plot.title = element_text(hjust = 0.5),
        text = element_text(size = 20))+
  geom_abline(intercept = 0, slope = 1, color="blue", linetype="dashed",size=2)
ggsave(p,width=9,height=7,file=paste0(name,'/scatter1.png'))
