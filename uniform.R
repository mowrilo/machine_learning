rm(list=ls())
setwd(dirname(sys.frame(1)$ofile))
library(MASS)

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

############################ Definição de centros ##################################################

centros_uniform <- expand.grid(seq(2,4,2),seq(2,4,2))
n_neurons <- nrow(centros_uniform)

plot(data[which(labels==1),],col='blue',xlim=c(-1,6),ylim=c(-1,6),ylab='',xlab='')
par(new=T)
plot(data[which(labels==-1),],col='red',xlim=c(-1,6),ylim=c(-1,6),ylab='',xlab='')
par(new=T)
plot(centros_uniform,col='green',xlim=c(-1,6),ylim=c(-1,6),ylab='',xlab='',pch=16)

############################### Treinamento da segunda camada #################################################

saidasrbf_uniform <- matrix(ncol=n_neurons,nrow=N)
for (i in 1:N){
    for (j in 1:n_neurons){
        saidasrbf_uniform[i,j] <- rbf_func(dataTrain[i,-ncol(dataTrain)],centros_uniform[j,],sigma)
    }
}
saidasrbf_uniform <- cbind(saidasrbf_uniform,1,dataTrain[,ncol(dataTrain)])

step <- .1
w_uniform <- matrix(runif(n_neurons+1),ncol=1)

itermax <- 200
for (iter in 1:itermax){
    trainingError <- 0
    saidasrbf_uniform <- saidasrbf_uniform[sample(N),]
    for (sample in 1:N){
        net <- as.numeric(saidasrbf_uniform[sample,-ncol(saidasrbf_uniform)] %*% w_uniform)
        y_uniform <- tanh(net)
        error <- saidasrbf_uniform[sample,ncol(saidasrbf_uniform)] - y_uniform
        trainingError <- trainingError + error^2
        w_uniform <- w_uniform + step*(1-tanh(net)^2)*error*saidasrbf_uniform[sample,-ncol(saidasrbf_uniform)]
    }
    cat(sprintf("Iter %d - Train MSE: %f\n",iter,trainingError/N))
}

saidastest_uniform <- matrix(ncol=n_neurons,nrow=nrow(dataTest))
for (i in 1:nrow(dataTest)){
    for (j in 1:n_neurons){
        saidastest_uniform[i,j] <- rbf_func(dataTest[i,-ncol(dataTrain)],centros_uniform[j,],sigma)
    }
}
saidastest_uniform <- cbind(saidastest_uniform,1,dataTest[,ncol(dataTest)])

resp_uniform <- tanh(saidastest_uniform[,-ncol(saidastest_uniform)] %*% w_uniform)
resp_uniform <- 2*(resp_uniform>0) - 1
erros <- sum(1*(resp_uniform != saidastest_uniform[,ncol(saidastest_uniform)]))/nrow(dataTest)
cat(sprintf("\nTest accuracy: %.2f \n",100*(1-erros)))
cat(sprintf("Number of neurons on hidden layer: %d\n",n_neurons))
