var numeric = require("./numeric.js");
var assert = require("assert");

const NUM_CATEGORIES = 9;
const testing_mode = true;
const training_rate = 0.001;

//biases are always the b argument
function add(a, b, testing_mode) {
	//assert that the second argument is a vector (i.e. the biases).
	if(testing_mode) {
		for(var check = 0; check < b.length; check++)
			assert(b[check].length === 1);
	}
	var height = a.length;
	var width = b.length;
	for(var i = 0; i < height; i++)
		for(var j = 0; j < width; j++)
			a[i][j] += b[i][0];
	return a;
};

function dot(a, b, testing_mode) {
	if(testing_mode)
		assert(a[0].length === b.length);
	return numeric.dot(a, b);
};

function mul(a, b, testing_mode) {
	if(testing_mode) {
		assert(a.length === b.length);
		assert(a[0].length === b[0].length);
	}
	return numeric.mul(a, b);
};

function sigmoid(z) {
	var e = numeric.exp(z);
	var denominator = numeric.add(1, e);
	return numeric.div(1, denominator);
};

function relu(z) {
	for(var i = 0; i < z.length; i++)
		for(var j = 0; j < z[0].length; j++)
			z[i][j] = Math.max(0, z[i][j]);
	return z;
};

function relu_prime(z) {
	for(var i = 0; i < z.length; i++)
		for(var j = 0; j < z[0].length; j++)
			z[i][j] = z[i][j] > 0 ? 1 : 0;
	return z;
};

function softmax(z) {
	for(var i = 0; i < z.length; i++) {
		var sum = 0;
		for(var j = 0; j < z[0].length; j++)
			sum += Math.exp(z[i][j]);
		for(var j = 0; j < z[0].length; j++)
			z[i][j] = Math.exp(z[i][j]) / sum;
	}
	return z;
};

function one_hot(y, testing_mode) {
	var one_hot = [];
	for(var i = 0; i < y.length; i++) {
		var position = [];
		for(var j = 0; j < NUM_CATEGORIES; j++)
			position[j] = 0;
		position[y[i]] = 1;
		one_hot.push(position);
	}
	if(testing_mode)
		assert(one_hot[0].length === NUM_CATEGORIES);
	return one_hot;
};

function cross_entropy(softmax, labels, testing_mode) {
	if(testing_mode) {
		assert(softmax.length === labels.length);
		assert(softmax[0].length === labels[0].length);
	}
	var ROW = softmax.length;
	var COL = softmax[0].length;
	var loss = 0;
	for(var i = 0; i < ROW; i++)
		for(var j = 0; j < COL; j++)
			loss += labels[i][j] * Math.log(softmax[i][j]);
	loss /= softmax.length;
	return -loss;
};

function cost(expected, actual) {

};

function back_prop(network, X, Y, g_prime, testing_mode, num_iter) {
	var D = [];

	for(var iter = 0; iter < num_iter; iter++) {
		//step 1: forward prop
		var expected = network.forward(X[iter]);
		var actual = Y[iter];

		var output_delta = [];
		//step 2: get delta between predictions and labels
		for(var j = 0; j < COL; j++)
			output_delta[j] = expected[j] - Y[j];
		console.log(dim(output_delta));

		if(testing_mode)
			assert(output_delta[0].length === expected[0].length && output_delta[0].length === actual[0].length);

		//step 3: get delta for the hidden layer
		var weight_contribs = dot(output_delta, h1.weights, testing_mode);
		var delta = mul(weight_contribs, g_prime(h1.inputs), testing_mode);

		//step 4: accumulate delta for hidden layer
		var delta_transpose = numeric.transpose(output_delta);
		console.log(dim(delta_transpose));
		console.log(dim(h1.activations));
		var gradients = dot(delta_transpose, h1.activations);
		console.log(dim(gradients));
		var m = expected.length;
		//step 5: divide accumulated gradients by m (number of training examples), add gradients to network.
		gradients = div(gradients, m);
	}


	return;
};

function dim(a) {
	return numeric.dim(a);
}

//recall that a data matrix X will be of dimensions (m x n), where you have m examples and n features.
//hence you have m rows, and n columns. 
var X = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [-1, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, -1]];

var Y = [4, 4, 4];

Y = one_hot(Y, testing_mode);

//weights for a given layer will be of dimensions (a x b), where the input is a matrix of (? x a). Output is of dimensions (? x b).
//? corresponds to number of examples, a corresponds to no. of nodes in previous layer, b corresponds to no. of nodes in this layer.
//the biases for this layer will be a vector of dimensions (b x 1). 
function HiddenLayer(dimensions, g) {
	this.weights = [];
	this.biases = [];
	var ROW = dimensions[0];
	var COL = dimensions[1];
	var epsilon = Math.sqrt(6) / (Math.sqrt(ROW + COL));
	//Formula for initialising weights: to be between -epsilon and epsilon, where epsilon is a function of the number of nodes
	//between the previous layer and the current layer.
	for(var i = 0; i < ROW; i++) {
		this.weights[i] = [];
		for(var j = 0; j < COL; j++)
			this.weights[i][j] = this.weights[i][j] = Math.random() * 2 * epsilon - epsilon;
	}
	//initialise biases to 0.
	for(var i = 0; i < ROW; i++) {
		this.biases[i] = [];
		this.biases[i][0] = 0;
	}
	this.g = g;
	this.inputs = [[]];
	this.activations = [[]];
};

HiddenLayer.prototype.forward = function(a, testing_mode) {
	var z = add(dot(a, this.weights, testing_mode), this.biases, testing_mode);
	this.inputs = numeric.clone(z);
	this.activations = this.g(z);
	return this.activations;
};

HiddenLayer.prototype.set_weights = function(W) {
	this.weights = W;
}

function OutputLayer(dimensions) {
	this.weights = [];
	this.biases = [];
	var ROW = dimensions[0];
	var COL = dimensions[1];
	var epsilon = Math.sqrt(6) / (Math.sqrt(ROW + NUM_CATEGORIES));
	//Formula for initialising weights: to be between -epsilon and epsilon, where epsilon is a function of the number of nodes
	//between the current layer and the next layer.
	for(var i = 0; i < ROW; i++) {
		this.weights[i] = [];
		for(var j = 0; j < COL; j++)
			this.weights[i][j] = Math.random() * 2 * epsilon - epsilon;
	}
	//initialise biases to 0.
	for(var i = 0; i < ROW; i++) {
		this.biases[i] = []
		this.biases[i][0] = 0;
	}
	this.logits = [];
};

OutputLayer.prototype.forward = function(a, testing_mode) {
	this.logits = add(dot(a, this.weights, testing_mode), this.biases, testing_mode);
	return this.logits;
};

OutputLayer.prototype.set_weights = function(W) {
	this.weights = W;
}

var h1 = new HiddenLayer([9, 9], relu);

var o = new OutputLayer([9, 9]);

var a = h1.forward(X, testing_mode);
var logits = o.forward(a, testing_mode);
var probabilities = softmax(logits);
var loss = cross_entropy(probabilities, Y, testing_mode);
console.log(loss);
back_prop(probabilities, Y, relu_prime, testing_mode, h1, 3);















