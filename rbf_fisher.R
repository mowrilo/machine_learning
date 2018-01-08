# Implementação do método apresentado em:
#		"RBF Neural Network Center Selection Based on 
#		Fisher Ratio Class Separability Measure", K. Z. Mao, 2002
# Autor: Murilo V. F. Menezes

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

candidates <- dataTrain[,-ncol(dataTrain)]

n_neurons <- nrow(candidates)
saidas <- matrix(nrow=N,ncol=n_neurons)
for (i in 1:N){ #Mapeia os neurônios
    for (j in 1:n_neurons){
        saidas[i,j] <- rbf_func(dataTrain[i,-ncol(dataTrain)],candidates[j,],sigma)
    }
}

labelsTrain <- dataTrain[,ncol(dataTrain)]
separabilities <- numeric()
fisher_sep <- numeric(ncol(saidas))

for (i in 1:ncol(saidas)){ # Calcula a separabilidade para cada neuronio candidato
    means_dif <- (mean(saidas[labelsTrain == 1,i]) - mean(saidas[labelsTrain == -1,i]))
    vars_sum  <- (var(saidas[labelsTrain == 1,i]) + var(saidas[labelsTrain == -1,i]))
    ratio <- means_dif^2/vars_sum
    fisher_sep[i] <- ratio
}

chosen <- which.max(fisher_sep)
separabilities <- c(separabilities,fisher_sep[chosen])
Q <- matrix(saidas[,chosen],ncol=1)
centros_fisher <- matrix(candidates[chosen,],nrow=1)
candidates <- candidates[-chosen,]
saidas <- saidas[,-chosen]

continue <- T
while (continue){
    #Ortogonaliza os vetores
    for (i in 1:ncol(saidas)){
        for (j in 1:ncol(Q)){
            costheta <- as.numeric(saidas[,i] %*% Q[,j])/(norm(saidas[,i])*norm(Q[,j]))
            saidas[,i] <- saidas[,i] - (saidas[,i]*costheta)
        }
    }
    
    saidas[which(is.nan(saidas))] <- 0
    
    fisher_sep <- numeric(ncol(saidas))
    for (i in 1:ncol(saidas)){
        means_dif <- (mean(saidas[labelsTrain == 1,i]) - mean(saidas[labelsTrain == -1,i]))
        vars_sum  <- (var(saidas[labelsTrain == 1,i]) + var(saidas[labelsTrain == -1,i]))
        ratio <- means_dif^2/vars_sum
        if (is.nan(vars_sum)){ratio <- 0} else if (vars_sum == 0){ratio <- 0}
        fisher_sep[i] <- ratio
    }
    
    chosen <- which.max(fisher_sep)
    separabilities <- c(separabilities,fisher_sep[chosen])
    Q <- cbind(Q,saidas[,chosen])
    centros_fisher <- rbind(centros_fisher,candidates[chosen,])
    candidates <- candidates[-chosen,]
    saidas <- saidas[,-chosen]
    
    plot(data[which(labels==1),],col='blue',xlim=c(-1,6),ylim=c(-1,6),ylab='',xlab='')
    par(new=T)
    plot(data[which(labels==-1),],col='red',xlim=c(-1,6),ylim=c(-1,6),ylab='',xlab='')
    par(new=T)
    plot(centros_fisher,col='green',xlim=c(-1,6),ylim=c(-1,6),ylab='',xlab='',pch=16)
    
    imprvmt <- fisher_sep[chosen]/sum(separabilities)
    cat(sprintf("Improvement on center #%d: %.2f\n",ncol(Q),100*imprvmt))
    if (imprvmt < .05){
        continue <- F
    }
}

n_neurons <- nrow(centros_fisher)

############################### Treinamento da segunda camada #################################################

saidasrbf_fisher <- matrix(ncol=n_neurons,nrow=N)
for (i in 1:N){
    for (j in 1:n_neurons){
        saidasrbf_fisher[i,j] <- rbf_func(dataTrain[i,-ncol(dataTrain)],centros_fisher[j,],sigma)
    }
}
saidasrbf_fisher <- cbind(saidasrbf_fisher,1,dataTrain[,ncol(dataTrain)])

step <- .1
w_fisher <- matrix(runif(n_neurons+1),ncol=1)

itermax <- 200
for (iter in 1:itermax){
    trainingError <- 0
    saidasrbf_fisher <- saidasrbf_fisher[sample(N),]
    for (sample in 1:N){
        net <- as.numeric(saidasrbf_fisher[sample,-ncol(saidasrbf_fisher)] %*% w_fisher)
        y_fisher <- tanh(net)
        error <- saidasrbf_fisher[sample,ncol(saidasrbf_fisher)] - y_fisher
        trainingError <- trainingError + error^2
        w_fisher <- w_fisher + step*(1-tanh(net)^2)*error*saidasrbf_fisher[sample,-ncol(saidasrbf_fisher)]
    }
    cat(sprintf("Iter %d - Train MSE: %f\n",iter,trainingError/N))
}

saidastest_fisher <- matrix(ncol=n_neurons,nrow=nrow(dataTest))
for (i in 1:nrow(dataTest)){
    for (j in 1:n_neurons){
        saidastest_fisher[i,j] <- rbf_func(dataTest[i,-ncol(dataTrain)],centros_fisher[j,],sigma)
    }
}
saidastest_fisher <- cbind(saidastest_fisher,1,dataTest[,ncol(dataTest)])

resp_fisher <- tanh(saidastest_fisher[,-ncol(saidastest_fisher)] %*% w_fisher)
resp_fisher <- 2*(resp_fisher>0) - 1
erros <- sum(1*(resp_fisher != saidastest_fisher[,ncol(saidastest_fisher)]))/nrow(dataTest)
cat(sprintf("\nTest accuracy: %.2f \n",100*(1-erros)))
cat(sprintf("Number of neurons on hidden layer: %d\n",n_neurons))
