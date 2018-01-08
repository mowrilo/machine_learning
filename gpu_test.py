import theano
import theano.tensor as T
import numpy as np
import time

x = T.matrix('x')
y = T.matrix('x')
z = T.dot(x,y)
f = theano.function([x,y],z,)#allow_input_downcast=True)

dim = 12000

X = np.random.rand(dim,dim).astype('float32')
print "N bytes: ",X.nbytes
Y = np.random.rand(dim,dim).astype('float32')

t1 = time.time()
Z = f(X,Y)
t2 = time.time()
print "Elapsed time: ", (t2-t1)
