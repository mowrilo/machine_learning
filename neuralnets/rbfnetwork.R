# Implementação de uma rede neural RBF, com seleção de centros via k-means
# Autor: Murilo V. F. Menezes

rm(list=ls())
library(MASS)
library(mlbench)

norm <- function(x){return(sqrt(sum(x^2)))}

rbf_func <- function(x,c,sig){
    return(exp(-1/2 * (norm(x-c)/(2*sig))^2))
}

set.seed(2)
data <- rbind(matrix(rnorm(500,2,.65),ncol=2),matrix(rnorm(500,4,.65),ncol=2))
labels <- c(rep(1,250),rep(-1,250))
data <- cbind(data,labels)
index <- sample(nrow(data))
dataTrain <- data[index[1:400],]
dataTest <- data[index[401:500],]

N <- nrow(dataTrain)

plot(data[which(labels==1),],col='blue',xlim=c(-1,6),ylim=c(-1,6),ylab='',xlab='')
par(new=T)
plot(data[which(labels==-1),],col='red',xlim=c(-1,6),ylim=c(-1,6),ylab='',xlab='')

sigma <- .6
k_kmeans <- 12

####################### Definição de centros #################################################

centros_kmeans <- kmeans(x = dataTrain[,-ncol(dataTrain)],centers = k_kmeans)$centers

######################### Treinamento da segunda camada ###################################### 

plot(data[which(labels==1),],col='blue',xlim=c(-1,6),ylim=c(-1,6),ylab='',xlab='')
par(new=T)
plot(data[which(labels==-1),],col='red',xlim=c(-1,6),ylim=c(-1,6),ylab='',xlab='')
par(new=T)
plot(centros_kmeans,col='green',xlim=c(-1,6),ylim=c(-1,6),ylab='',xlab='',pch=16)

saidasrbf_kmeans <- matrix(ncol=k_kmeans,nrow=N)
for (i in 1:N){
    for (j in 1:k_kmeans){
        saidasrbf_kmeans[i,j] <- rbf_func(dataTrain[i,-ncol(dataTrain)],centros_kmeans[j,],sigma)
    }
}
saidasrbf_kmeans <- cbind(saidasrbf_kmeans,1,dataTrain[,ncol(dataTrain)])

step <- .1
w_kmeans <- matrix(runif(k_kmeans+1),ncol=1)

itermax <- 200
for (iter in 1:itermax){
    trainingError <- 0
    saidasrbf_kmeans <- saidasrbf_kmeans[sample(N),]
    for (sample in 1:N){
        net <- as.numeric(saidasrbf_kmeans[sample,-ncol(saidasrbf_kmeans)] %*% w_kmeans)
        y_kmeans <- tanh(net)
        error <- saidasrbf_kmeans[sample,ncol(saidasrbf_kmeans)] - y_kmeans
        trainingError <- trainingError + error^2
        w_kmeans <- w_kmeans + step*(1-tanh(net)^2)*error*saidasrbf_kmeans[sample,-ncol(saidasrbf_kmeans)]
    }
    cat(sprintf("Iter %d - Train MSE: %f\n",iter,trainingError/N))
}

saidastest_kmeans <- matrix(ncol=k_kmeans,nrow=nrow(dataTest))
for (i in 1:nrow(dataTest)){
    for (j in 1:k_kmeans){
        saidastest_kmeans[i,j] <- rbf_func(dataTest[i,-ncol(dataTrain)],centros_kmeans[j,],sigma)
    }
}
saidastest_kmeans <- cbind(saidastest_kmeans,1,dataTest[,ncol(dataTest)])

resp_kmeans <- tanh(saidastest_kmeans[,-ncol(saidastest_kmeans)] %*% w_kmeans)
resp_kmeans <- 2*(resp_kmeans>0) - 1
erros <- sum(1*(resp_kmeans != saidastest_kmeans[,ncol(saidastest_kmeans)]))/nrow(dataTest)
cat(sprintf("Test accuracy: %.2f\n",100*(1-erros)))
