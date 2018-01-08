# Implementação do método de memória matricial
# OLAM (Optimal Linear Adaptative Memory),
# proposto por Teuvo Kohonen e Matti Ruohonen

rm(list=ls())

library(MASS)

#Inicializa-se dois pares de relações 
x1 <- c(0,-0.15,-0.29,0.88,-0.29,-0.15,0,0,0,0)
y1 <- c(1,1,0,1,0,0,1,0,0,0)
x2 <- c(0,0,-0.06,-0.09,-0.40,0,0.89,0,-0.3,-0.15)
y2 <- c(0,0,0,1,1,1,0,0,0,0)

#Normalizando
x1 <- x1/sqrt(sum(x1^2))
x2 <- x2/sqrt(sum(x2^2))

#Agrupa-se em matrizes
X <- cbind(x1,x2)
Y <- cbind(y1,y2)

#Calcula-se a matriz W pela regra de Hebb
W_hebb <- Y %*% t(X)
#Calcula-se a matriz W pelo método OLAM
W_olam <- Y %*% ginv(X)

#Recuperando os vetores y pela regra de Hebb
y1_rec_hebb <- W_hebb %*% x1
cbind(y1,y1_rec_hebb)
y2_rec_hebb <- W_hebb %*% x2
cbind(y2,y2_rec_hebb)

#Recuperando os vetores y pelo método OLAM
y1_rec_olam <- W_olam %*% x1
cbind(y1,y1_rec_olam)
y2_rec_olam <- W_olam %*% x2
cbind(y2,y2_rec_olam)

#Exemplo da regra de Hebb com vetores ortogonais mas não-normalizados
x1 <- c(0.3, 0.2)
y1 <- 0.9
x2 <- c(-0.2, 0.3)
y2 <- 0.5

#Agrupa-se em matrizes
X <- cbind(x1,x2)
Y <- cbind(y1,y2)

#Calcula-se W e recupera ys
W_hebb <- Y %*% t(X)
y1_rec_hebb <- W_hebb %*% x1
cbind(y1,y1_rec_hebb)
y2_rec_hebb <- W_hebb %*% x2
cbind(y2,y2_rec_hebb)

#Valores são multiplicados pelos produtos internos
y1 * (x1 %*% x1)
y2 * (x2 %*% x2)

#Com o OLAM
W_olam <- Y %*% ginv(X)

y1_rec_olam <- W_olam %*% x1
cbind(y1,y1_rec_olam)
y2_rec_olam <- W_olam %*% x2
cbind(y2,y2_rec_olam)


#Normaliza-se
x1 <- x1/sqrt(sum(x1^2))
x2 <- x2/sqrt(sum(x2^2))

X <- cbind(x1,x2)
Y <- cbind(y1,y2)

W_hebb <- Y %*% t(X)
y1_rec_hebb <- W_hebb %*% x1
cbind(y1,y1_rec_hebb)
y2_rec_hebb <- W_hebb %*% x2
cbind(y2,y2_rec_hebb)

#Por fim, calculando o OLAM com vetores linearmente dependentes
x1 <- c(-0.4, -0.6)
y1 <- 0.9
x2 <- c(0.2, 0.3)
y2 <- 0.5

X <- cbind(x1,x2)
Y <- cbind(y1,y2)

W_olam <- Y %*% ginv(X)
y1_rec_olam <- W_olam %*% x1
cbind(y1,y1_rec_olam)
y2_rec_olam <- W_olam %*% x2
cbind(y2,y2_rec_olam)

W_hebb <- Y %*% t(X)
y1_rec_hebb <- W_hebb %*% x1
cbind(y1,y1_rec_hebb)
y2_rec_hebb <- W_hebb %*% x2
cbind(y2,y2_rec_hebb)
