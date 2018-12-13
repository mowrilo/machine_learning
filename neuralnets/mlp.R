# an MLP with backpropagation method. 
# can be used for both classification or regression.

rm(list=ls())
setwd(dirname(sys.frame(1)$ofile))

library(plot3D)
library(rgl)
library(mlbench)

outputLayer <- "nlinear"

der_tanh <- function(x){
  return(1 - tanh(x)^2)
  # return(cos(x))
}

ident <- function(x){
  return(x)
}

der_lin <- function(x){
  return(1)
}

if (outputLayer == "nlinear"){
  der <- der_tanh
  finalFunc <- tanh
} else{
  der <- der_lin
  finalFunc <- ident
}
# x <- seq(0,10,.01)
# x <- sample(x)
# y <- sin(2*x)/exp(x/5) + rnorm(length(x),0,.2)

p <- mlbench.2dnormals(100,sd = 10)#,.5,.08)
plot(p)
x <- p[[1]]
y  <- as.numeric(p[[2]])-1
y2 <- sample(y)
y <- y2

dataAll <- cbind(1,x,y)

data <- dataAll#[1:500,]
# dataTest <- dataAll[501:750,]
# dataVal <- dataAll[751:nrow(dataAll),]

hiddens <- c(4)
nNeur <- c(50)
lambda <- 0
maxcount <- 5000
eta <- 1e-3

means <- numeric(maxcount)
sds <- numeric(maxcount)

mses <- matrix(nrow=length(hiddens),ncol=length(nNeur))
ch <- 0
for (nhid in hiddens){
  ch <- ch+1
  cn <- 0
  for (nn in nNeur){
    cn <- cn+1
    cat(sprintf("Number of hidden layers: %d\nNumber of neurons on each layer: %d",
                nhid,nn))
    w <- list()
    nhidden <- rep(nn,nhid)
    nhidlayers <- length(nhidden)
    set.seed(5)
    for (i in 1:nhidlayers){
      if (i ==1){
        w[[i]] <- matrix(runif(nhidden[i]*(ncol(data)-1),min = -.5,max=.5),nrow=nhidden[i],ncol=ncol(data)-1)
      } else{
        w[[i]] <- matrix(runif(nhidden[i]*(nhidden[i-1]+1),min = -.5,max=.5),nrow=nhidden[i],ncol=nhidden[i-1]+1)
      }
    }
    w[[nhidlayers+1]] <- matrix(runif(nhidden[nhidlayers]+1,min = -.5,max=.5),nrow=1,ncol=nhidden[nhidlayers]+1)
    
    count <- 0
    error <- c(Inf)
    ci <- 1
    cont <- T
    
    while ((count < maxcount) & (cont)){
      count <- count+1
      data <- data[sample(nrow(data)),]
      error2 <- 0
      cat(sprintf("\nEpoch: %d",count))
      
      mn <- 0
      s <- 0
      
      for (i in 1:nrow(data)){
        # Propagating
        datum <- list(as.matrix(data[i,-ncol(data)]))
        for (j in 1:(length(w)-1)){
          datum[[j+1]] <- as.matrix(c(1,tanh(w[[j]] %*% datum[[j]])))
        }
        datum[[length(w)+1]] <- finalFunc(w[[length(w)]] %*% datum[[length(w)]])
        if (any(is.nan(datum[[j+1]]))){
          break()
        }
        resp <- datum[[length(datum)]]
        norma <- 0
        for (asd in 1:length(w)){
            norma <- norma + sqrt(sum(w[[asd]]^2))
        }
        er1 <- (data[i,ncol(data)] - resp)
        erro <- er1 + sign(er1)*lambda*norma
        # cat(sprintf("Resp: %f\nTrue: %f\nErro: %f\n\n",resp,data[i,ncol(data)],erro))
        error2 <- error2 + erro^2
        # print("valores antes:")
        # print(erro)
        # print(datum[[nhidlayers+1]])
        # print(w[[nhidlayers+1]])
        delta <- list()
        # print("net on output:")
        # print(datum[[nhidlayers+1]] %*% t(w[[nhidlayers+1]]))
        delta[[nhidlayers+1]] <- erro*der(w[[length(w)]] %*% datum[[length(w)]])
        # print("delta antes:")
        # print(delta)
        
        for (j in length(w):1){
          w[[j]] <- w[[j]] + eta* (delta[[j]] %*% t(datum[[j]]))
          if (j==2){
              mn <- mn + mean(abs((delta[[2]] %*% t(datum[[2]]))))
              s <- s + sd(abs((delta[[2]] %*% t(datum[[2]]))))
          }
          if (j != 1){
            a <- as.matrix(w[[j]][,-1])
            if (ncol(a) != 1){
              a <- t(a)
            }
            delta[[j-1]] <- (a %*% delta[[j]]) * der_tanh(w[[j-1]] %*% datum[[j-1]])
          }
        }
      }
      means[count] <- mn/maxcount
      sds[count] <- s/maxcount
      ci <- ci+1
      error2 <- error2/nrow(data)
      cat(sprintf("\tError: %.4f",error2))
      error <- c(error,error2)
      # if (abs(error2) < .02){
      #   cont <- F
      # }
      # if (!(count %% 100)){
      #   plot(error,type='l',xlim=c(0,maxcount),xlab='epoch')
      # }
    }
    # mse <- 0
    # for (i in 1:nrow(dataVal)){
    #   dados <- list(as.matrix(dataVal[i,-ncol(dataVal)]))
    #   for (j in 1:(length(w)-1)){
    #     dados[[j+1]] <- as.matrix(c(1,tanh(w[[j]] %*% dados[[j]])))
    #   }
    #   resp <- finalFunc(w[[length(w)]] %*% dados[[length(w)]])
    #   mse <- mse + (dataVal[i,ncol(dataVal)] - resp)^2
    # }
    # 
    # mse <- mse/nrow(dataVal)
    # mses[ch,cn] <- mse
    # cat(sprintf("MSE: %f\n\n",mse))
  }
}

plot(1:length(error),error,'l')

# par(mfrow=c(1,2))
# plot(log10(1:maxcount),means,'l')
# par(new=T)
# plot(log10(1:maxcount),sds,'l',col='red')
# nhidlayers <- 3

# v <- x
# ytest <- y

# 
ptest <- mlbench.2dnormals(100,sd=10)
v <- ptest[[1]]#seq(0,100,.01)
ytest <- as.numeric(ptest[[2]])-1
r <- numeric(nrow(v))

# ci <- 0
for (ci in 1:nrow(v)){
  # ci <- ci+1
    xx <- v[ci,]
  dados <- list(c(1,xx))
  for (j in 1:(length(w)-1)){
    dados[[j+1]] <- as.matrix(c(1,tanh(w[[j]] %*% dados[[j]])))
  }
  dados[[length(w)+1]] <- finalFunc(w[[length(w)]] %*% dados[[length(w)]])
  r[ci] <- dados[[length(w)+1]][1]
}

r <- 1*(r > .5)
acc <- sum(r == ytest)/100
print(acc)

# 
# plot(x,y,ylim=c(-1.5,1.5))
# par(new=T)
# plot(v,r,'l',ylim=c(-1.5,1.5),col='red')
# 
# 
