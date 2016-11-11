setwd('/home/chuna/TNT-release/record_data'); # To file location
library(ggplot2)
library(R.matlab)
library(Rtsne)

source('util.R')

####################
# Read data
####################
prefix    = 'TNTforHDA';
S_dataset = 'CALTECH';
T_dataset = 'CALTECH';
filename  = max(list.files(path = ".", pattern = sprintf('%s_%sto%s_*',prefix,S_dataset,T_dataset)));
data_in   = readMat(filename,fixNames = FALSE);
data_in   = lapply(data_in,ListArr2List);

label_names = c('backpack','bike','calculator','headphones','keyboard',
                'laptop','monitor','mouse','mug','projector');

all_data   = do.call(cbind,lapply(data_in,function(d) d$projected_data));
all_label  = do.call(c,    lapply(data_in,function(d) d$label));
all_domain = do.call(c,    mapply(function(d,n) rep(n,times = length(d$label)) , data_in, as.list(names(data_in)) ))

####################
# T-SNE
####################
tsne = Rtsne(t(all_data), check_duplicates = FALSE, pca = TRUE, perplexity=10, theta=0.5, dims=2);

embedding = data.frame(
              data   = tsne$Y,
              class  = as.factor( t(label_names[all_label])),
              domain = all_domain
            );
embedding$domain = factor(embedding$domain,levels(embedding$domain)[c(2,1,3)])

####################
# Plot
####################
plot_noweight = function(embedding){
  ggplot(embedding, aes(x=data.1, y=data.2, color= class))+
    geom_point(aes(size = domain,shape = domain ))+
    scale_size_manual(values = c(6,6,3))+
    scale_shape_manual(values = c(3,16,16))+
    guides(colour = guide_legend(override.aes = list(size=6))) +
    xlab("") + ylab("") +
    theme_light(base_size=20)+ 
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          #legend.position=c(0.9,0.7),
          strip.background = element_blank(),
          strip.text.x     = element_blank(),
          axis.text.x      = element_blank(),
          axis.text.y      = element_blank(),
          axis.ticks       = element_blank(),
          axis.line        = element_blank(),
          panel.border     = element_blank())
}
p = plot_noweight(embedding)
print(p)
#ggsave("C_noprune_small.png", p, width=6, height=4, units="in")