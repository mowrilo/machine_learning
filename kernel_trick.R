# Visualization of the Kernel Trick in SVMs, using the polynomial kernel
# Murilo Menezes

p <- mlbench.circle(300)

data <- p[[1]]
labels <- p[[2]]

df <- data.frame(cbind(data,labels))
df$labels <- as.factor(df$labels)
ggplot(df,aes(x=V1,y=V2,colour=labels)) + geom_point()

# linear SVM on the input space
model <- svm(data,labels,type="C-classification",
             kernel="linear")
pred <- predict(model, data)
sum(pred == labels)/nrow(data)
# there is no hyperplane that separates the classes decently!

k <- function(x1,x2){
    return((x1 %*% x2)^2)
}

# explicitly mapping the samples...
phi <- function(x){
    return(c(x[1]^2,x[2]^2,sqrt(2)*x[1]*x[2]))
}

data_2 <- matrix(nrow=nrow(data),ncol=3)
for (i in 1:nrow(data)){
    data_2[i,] <- phi(data[i,])
}

library(plotly)
plot_ly(x=data_2[,1],y=data_2[,2],z=data_2[,3],type='scatter3d',
        mode='markers',color=labels)

# the same linear SVM...
model <- svm(data_2,labels,type="C-classification",
             kernel="linear")
pred <- predict(model, data_2)
sum(pred == labels)/nrow(data)
# now it can separate the classes!