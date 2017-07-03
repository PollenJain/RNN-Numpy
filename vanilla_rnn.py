import numpy as np

file_handler = open('pos.txt', 'r')

file_content_as_a_string = file_handler.read()

unique_characters = set(file_content_as_a_string)

list_of_unique_characters = list(unique_characters)

print("list of unique characters " , list_of_unique_characters)

length_of_document = len(file_content_as_a_string)

print("length of document", length_of_document)

vocab_size = len(list_of_unique_characters) # Similar to width x height of an image
print("vocab size", vocab_size)

''' character to index encoding and vice versa'''
character_to_index_mapping = { ch:i for i,ch in enumerate(list_of_unique_characters) }

index_to_character_mapping = { i:ch for i,ch in enumerate(list_of_unique_characters) }

# hyperparameters
hidden_size = 100 # size of the hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for, similar to batch size
learning_rate = 1e-1

# model parameters
U = np.random.randn(hidden_size, vocab_size)* 0.01 # (100, vocab_size), input to hidden, Ux
W = np.random.randn(hidden_size, hidden_size)*0.01 # (100, 100), hidden to (next) hidden
V = np.random.randn(vocab_size, hidden_size)*0.01 # (vocab_size, 100), hidden to output

bh = np.zeros((hidden_size,1)) # (100, 1), hidden bias
by = np.zeros((vocab_size,1))  # (vocab_size, 1), output bias

 
def lossFun(inputs, targets, hprev):
	''' inputs : list of integers, [ x0, x1, x2, ... , xt ] where xt is the input at time t. For ex: x0 is input at time t=0, x4 is input at time t=4. Here x0, x1, ..., xt are not one-hot encoded.
	    len(inputs) = seq_length
	    len(targets) = seq_length
	    Note: seq_length is different from vocab_size. vocab_size is no of unique characters in the entire document.
	    In terms of images, seq_length is same as batch size and vocab_size is no of pixels in one image.
	    Each character xi when one hot encoded can be seen as one flattened image.
	    targets : list of integers, [ t0, t1, t2, ... , tt ] where t0 should be equal to x1, targets => [ x1, x2, x3, ... , x(t+1) ] 
	    hprev : Hx1 numpy array of initial hidden state, this is considered as the hidden state at time t=-1. Note we do not have x corresponding to -1 in inputs list.


	   returns :
		loss,
		gradients on model parameters,
		last hidden state

	
	'''

	hs = dict()
	hs[-1] = np.copy(hprev) # Since for calculating hidden state at time t=0 we require hidden state at time t=-1
	loss = 0


	xs = dict()
	ys = dict()
	ps = dict()

	''' forward pass '''
	for t in range(len(inputs)):
		if t not in xs:
			xs[t] = np.zeros((vocab_size,1)) # vocab is similar to alphabet of TOC, encode in 1-of-k representation, same as 1 hot encoding, xs[t] is a 2-dimensional numpy array, xs is a 								 # dictionary
		xs[t][inputs[t],0] = 1 # one-hot encoding inputs, assuming vocab_size = 5, then xs[0] = np.array([ [ 1 ], [ 0 ], [ 0 ], [ 0 ], [ 0 ] ] ])

		# Dimensional analysis: U: (100, 5) xs[0]: (5, 1), then we get (100, 1)
		#			W: (100, 100) hs[t-1]: (100, 1), then we get (100, 1)
		#			bh:(100, 1)
		#			hs[0] => (100, 1), just like hs[-1]

		hs[t] = np.tanh ( np.dot(U, xs[t]) + np.dot(W, hs[t-1]) + bh) # hidden state at time t, is a function of previous hidden state and current input.

		# Dimensional analysis: V: (5, 100) hs[0]: (100, 1), then we get (5, 1)
		#			by:(5, 1)
		#			ys[0] => (5, 1), just like xs[0]
		ys[t] = np.dot( V, hs[t] ) + by # Output at time t is purely a function of the hidden state at time t

		# similarly, ps[0] is of dimension (5, 1), a vector containing probablities for next characters, ps[0] is a 2-dimensional numpy array
		# Since, y[t] itself is a 2-dimensional numpy array, so
		# For ex: y[0] = [1, 2, 3, 4, 5]
		# Then, p[0] = [p1, p2, p3, p4, p5] 
		# where, p1 = exp(1) / exp(1) + exp(2) + exp(3) + exp(4) + exp(5)
		#	 p2 = exp(2) / exp(1) + exp(2) + exp(3) + exp(4) + exp(5)
		#	 p3 = exp(3) / exp(1) + exp(2) + exp(3) + exp(4) + exp(5)
		# 	 p4 = exp(4) / exp(1) + exp(2) + exp(3) + exp(4) + exp(5)
		# 	 p5 = exp(5) / exp(1) + exp(2) + exp(3) + exp(4) + exp(5)

		ps[t] = np.exp( ys[t] ) / np.sum( np.exp( ys[t] ) )# ProbablitieS (note its not just A probablity ) for next characterS ( note its not just A character )
		
		# targets[0] should be probablity for x1 given input x0		
		loss = loss - np.log( ps[t][targets[t],0] ) # softmax ( cross-entropy loss ), for xs[0], loss = log ( probablity for x1 ) since target[0] = x1

		
	''' backward pass '''


	dU = np.zeros((hidden_size, vocab_size))
	dW = np.zeros((hidden_size, hidden_size))
	dV = np.zeros((vocab_size, hidden_size))
				
	dbh = np.zeros((hidden_size, 1))
	dby = np.zeros((vocab_size, 1))

	dhnext = np.zeros((hidden_size, 1))

	# same as reversed(range(len(inputs)))
	for t in range(len(inputs)-1, -1, -1):  # assuming vocab_size = 5
		# ps[4] is a 2-dimensional vector, so dy is a 2-dimensional vector
		dy = np.copy(ps[t]) 
		dy[targets[t], 0] = dy[targets[t],0] - 1
		dV = dV + np.dot(dy, hs[t].T)
		dby = dby + dy
		dh = np.dot(V.T, dy) # Note that even h is updated in RNN, backprop into h.
		dhraw = ( 1 - hs[t] * hs[t] ) * dh # backprop through tanh non-linearity
		dbh = dbh + dhraw
		dU = dU + np.dot(dhraw, xs[t].T)
		dW = dW + np.dot(dhraw, hs[t-1].T)
		dhnext = np.dot(W.T, dhraw)

	for dparam in [dU, dW, dV, dbh, dby]:
		np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
	
	return loss, dU, dW, dV, dbh, dby, hs[len(inputs)-1]

	
def sample(h, index_of_a_character_at_time_t, n):
	''' sample a sequence of n integers from the model
	h is memory state at time (t-1) : h (hidden_size,1),
	index_of_a_character_at_time_t,
	we want to generate next n characters starting with index_of_a_character_at_time_t'''

	global vocab_size
	
	x = np.zeros((vocab_size, 1)) # Converting index_of_a_character_at_time_t to one-hot encoded vector, x is a 2-dimensional vector with shape (5,1)
	x[index_of_a_character_at_time_t] = 1
	indices_of_character_in_sequence = []

	# Assuming hidden_size = 100, vocab_size = 5
	for t in range(n): # Given, index_of_a_character_at_time_t, we want to generate next n characters => we can say that given one character we want to generate the entire story.
		# Dimensional Analysis : U : (100, 5)
		#			 x : (5, 1)
		#			 Ux : (100, 1)
		#			 W : (100, 100)
		#			 h : (100, 1)
		#			 Wh : (100, 1)
		#			 bh : (100, 1)
		#			 h = np.tanh( Ux + Wh + bh ) => (100, 1)
		h = np.tanh ( np.dot(U, x) + np.dot(W, h) + bh ) # using the learnt U, W, bh so far

		# Dimensional Analysis : V : (5, 100)
		#			 h : (100, 1)
		#			Vh : (5, 1)
		#			by : (5, 1)
		#			y : (5, 1)
		y = np.dot( V, h ) + by # using the learnt V, by so far

		# p has same shape as y => (5, 1)
		p = np.exp(y) / sum(np.exp(y)) # what should be the next character given index_of_a_character_at_time_t is determined by the maximum probablity
		# Out of 5 index, randomly choose ONE of the index based on the probablity distribution p. Its not necessary that the index with maximum probablity will only come.
		# But for sure, the index with corresponding probablity of  0 will not be returned.
		ix = np.random.choice(range(vocab_size), p = p.ravel() ) # p.ravel() converts p from (5,1) to (5,) 
		x = np.zeros((vocab_size,1)) 
		x[ix] = 1 # one hot encode the new character (x(t+1)) and in the next iteration find the next possible character ( x(t+2) )
		indices_of_character_in_sequence.append(ix)

	return indices_of_character_in_sequence
		

i, start = 0, 0

mU = np.zeros((hidden_size, vocab_size))
mW = np.zeros((hidden_size, hidden_size))
mV = np.zeros((vocab_size, hidden_size))

mbh = np.zeros((hidden_size,1))
mby = np.zeros((vocab_size,1))

''' Note: RNN has two kind of loss.
	1. Which we updated in the function named lossFun.
	2. And smooth loss. This is purely a function of vocab_size and seq_length
'''

smooth_loss = -np.log(1.0/ vocab_size)*seq_length; 

# max no of iterations = infinity
while True:	
	
	# if next batch will exceed the size of the file or if this is our first iteration 
	if start+seq_length+1 >= len(file_content_as_a_string) or i == 0:
		hprev = np.zeros((hidden_size,1)) # reset RNN Memory in every iteration. Initialize it with zeros.
		start = 0 # read from the beginning of the file

	# Each batch consists of seq_length characters. 
	# Here, each batch consists of 25 characters.
	# Input is a list containing index mapping of each of these characters.
	inputs = [character_to_index_mapping[ch] for ch in file_content_as_a_string[start:start+seq_length]] # inputs = [28, 14, 3, ..., 52], len(inputs) is equal to seq_length
	targets = [character_to_index_mapping[ch] for ch in file_content_as_a_string[start+1:start+1+seq_length]] # targets = [14, 3, ..., 29 ], len(targets) is equal to seq_length

	if i%100 == 0:
		indices_of_character_in_sequence_given_character_at_index_zero = sample(hprev, inputs[0], 200) # 200 characters predicted by our model
		generated_text = ''.join(index_to_character_mapping[ix] for ix in indices_of_character_in_sequence_given_character_at_index_zero)
		print(" ---------Generated txt at iteration number ", i, " -----------------")
		print(generated_text)

	loss, dU, dW, dV, dbh, dby, hprev = lossFun(inputs, targets, hprev)
	
	# Updating smooth loss
	smooth_loss = smooth_loss * 0.999 + loss * 0.001

	if i%100 == 0:
		print(" Loss: " , loss )


	# perform parameter upgrade with Adagrad
	for param, dparam, mem in zip( [ U, W, V, bh, by ], [ dU, dW, dV, dbh, dby ], [ mU, mW, mV, mbh, mby ] ):
		mem = mem + (dparam*dparam)
		param = param -  (learning_rate*dparam)/ np.sqrt(mem + 1e-8) # adagrad update

	start = start + seq_length # updating to get next set of seq_length characters
	i = i + 1 # Keeping track of number of iterations

	
