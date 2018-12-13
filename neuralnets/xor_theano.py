import theano
import theano.tensor as T
import numpy as np
import time

x = T.vector('x',dtype='float32')
y = T.scalar('y',dtype='float32')

Xdata = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
Ydata = np.array([0,1,1,0]).astype('float32')
n_epochs = 10000

np.random.seed(2)
w1 = theano.shared(np.random.randn(2,2).astype('float32'),name="w1")
b1 = theano.shared(np.random.randn(2).astype('float32'),name="b1")
w2 = theano.shared(np.random.randn(1,2).astype('float32'),name="w2")
b2 = theano.shared(np.random.randn(1).astype('float32'),name="b2")

net1 = 1/(1+T.exp(-(T.dot(w1,x) + b1)))
out = 1/(1+T.exp(-(T.dot(w2,net1) + b2)))
pred = out.sum() > .5
cost = .5 * (y - out.sum())**2

gw1,gb1,gw2,gb2 = T.grad(cost,[w1,b1,w2,b2])

eta = .1

train = theano.function(inputs=[x,y],outputs=[pred,cost],updates=((w1,w1 - eta*gw1),(w2,w2 - eta*gw2),(b1,b1 - eta*gb1),(b2,b2 - eta*gb2)))
predict = theano.function(inputs=[x],outputs=pred)

print "Training..."

t1 = time.time()
for count in xrange(n_epochs):
    for i in xrange(Xdata.shape[0]):
        pred, error = train(Xdata[i],Ydata[i])
        #print Xdata[i],pred,error
t2 = time.time()
print "Elapsed time: ",(t2-t1)

print "Final weights and biases of input to hidden layer: \n",w1.get_value(),b1.get_value()
print "Final weights and biases of hidden to output layer: \n",w2.get_value(),b2.get_value()

print "Testing..."

preds = np.array([])
for i in xrange(Xdata.shape[0]):
    preds = np.append(preds,predict(Xdata[i]))

print "Final training cost: ",error
print "Targets: ",Ydata
print "Predictions: ",preds
