package neuralNetwork

import (
	"fmt"
	"math"
	"strconv"
	"time"

	ngo "deep_learning/numeric"

	"gonum.org/v1/gonum/mat"
)

type neuralNetwork struct {
	nn_structure   []int
	num_iterations int
	learning_rate  float64
	activation     activation
	parameters     map[string]*mat.Dense
}

func NewNeuralNetwork(nn_structure []int, actOpt actType, num_iterations int, learning_rate float64) neuralNetwork {

	act := activation{}
	switch actOpt {
	case ActivationTanh:
		act = activation{function: tanhActivation, derivative: tanhPrimeActivation}
	case ActivationSigmoid:
		act = activation{function: sigmoidActivation, derivative: sigmoidPrimeActivation}
	case ActivationElu:
		act = activation{function: eluActivation, derivative: eluPrimeActivation}
	}

	return neuralNetwork{
		nn_structure:   nn_structure,
		num_iterations: num_iterations,
		learning_rate:  learning_rate,
		activation:     act,
	}
}

// initializing the model parameters
func initializeParameters(nn_structure []int) map[string]*mat.Dense {
	parameters := make(map[string]*mat.Dense) // map containing the parameters
	L := len(nn_structure) - 1                // number of layers

	for l := 0; l < L; l++ {
		scalar := math.Sqrt((6.0 / float64(nn_structure[l]+nn_structure[l+1])))

		parameters["W"+strconv.Itoa(l+1)] = ngo.Scale(scalar, ngo.Randn(nn_structure[l+1], nn_structure[l]))
		parameters["b"+strconv.Itoa(l+1)] = mat.NewDense(nn_structure[l+1], 1, nil)
	}

	return parameters
}

// forward propagation step
func forwardPropagation(parameters map[string]*mat.Dense, x *mat.Dense, actFunc activationFunction) (*mat.Dense, map[string]*mat.Dense, map[string]*mat.Dense) {
	L := len(parameters) / 2         // number of layers
	Z := make(map[string]*mat.Dense) // linear function
	A := make(map[string]*mat.Dense) // activation function
	A[strconv.Itoa(0)] = x

	for l := 0; l < L; l++ {
		W := parameters["W"+strconv.Itoa(l+1)] // weights W
		b := parameters["b"+strconv.Itoa(l+1)] // biases b

		applyActFunction := func(_, _ int, v float64) float64 { return actFunc(v) }

		Z[strconv.Itoa(l+1)] = ngo.AddMatrixVector(ngo.MatMul(W, A[strconv.Itoa(l)]), b) // compute the linear operation
		A[strconv.Itoa(l+1)] = ngo.Apply(applyActFunction, Z[strconv.Itoa(l+1)])         // compute the non linear operation
	}

	// prediction
	y_hat := Z[strconv.Itoa(L)]

	return y_hat, Z, A
}

// computing the cost function
func costFunction(y_hat, y *mat.Dense) float64 {
	m := y_hat.RawMatrix().Cols
	sum := mat.Sum(ngo.Square(ngo.Sub(y_hat, y)))

	return (1.0 / (2.0 * float64(m)) * sum)
}

// backward propagation step
func backwardPropagation(parameters, Z, A map[string]*mat.Dense, y, y_hat *mat.Dense, actPrimeFunc activationFunction) (map[string]*mat.Dense, map[string]*mat.Dense) {
	m := y.RawMatrix().Cols  // number of training examples
	L := len(parameters) / 2 // number of layers

	dZ := make(map[string]*mat.Dense) // derivatives of the linear function Z
	dW := make(map[string]*mat.Dense) // derivatives of the weigths W
	db := make(map[string]*mat.Dense) // derivatives of the biases b
	dA := make(map[string]*mat.Dense) // derivatives of the activation function A

	dZ[strconv.Itoa(L)] = ngo.Scale(1./float64(m), ngo.Sub(y_hat, y))
	dW[strconv.Itoa(L)] = ngo.MatMul(dZ[strconv.Itoa(L)], A[strconv.Itoa(L-1)].T())
	db[strconv.Itoa(L)] = ngo.SumRows(dZ[strconv.Itoa(L)])

	for l := L - 1; l > 0; l-- {
		applyActPrimeFunction := func(_, _ int, v float64) float64 { return actPrimeFunc(v) }

		dA[strconv.Itoa(l)] = ngo.MatMul(parameters["W"+strconv.Itoa(l+1)].T(), dZ[strconv.Itoa(l+1)])
		dZ[strconv.Itoa(l)] = ngo.Multiply(dA[strconv.Itoa(l)], ngo.Apply(applyActPrimeFunction, Z[strconv.Itoa(l)]))
		dW[strconv.Itoa(l)] = ngo.MatMul(dZ[strconv.Itoa(l)], A[strconv.Itoa(l-1)].T())
		db[strconv.Itoa(l)] = ngo.SumRows(dZ[strconv.Itoa(l)])
	}

	return dW, db
}

// update the parameters (gradient descent)
func updateParameters(parameters, dW, db map[string]*mat.Dense, learning_rate float64) map[string]*mat.Dense {
	L := len(parameters) / 2 // number of layers

	for l := 0; l < L; l++ {
		parameters["W"+strconv.Itoa(l+1)] = ngo.Sub(parameters["W"+strconv.Itoa(l+1)], ngo.Scale(learning_rate, dW[strconv.Itoa(l+1)]))
		parameters["b"+strconv.Itoa(l+1)] = ngo.Sub(parameters["b"+strconv.Itoa(l+1)], ngo.Scale(learning_rate, db[strconv.Itoa(l+1)]))
	}

	return parameters
}

// train model
func (network *neuralNetwork) Fit(x_train, y_train *mat.Dense, print_cost bool) []float64 {

	start := time.Now()

	// keep track of the cost
	costs := []float64{}

	// initializing the model parameters
	network.parameters = initializeParameters(network.nn_structure)

	// loop
	for i := 0; i < network.num_iterations; i++ {
		// forward propagation
		y_hat, Z, A := forwardPropagation(network.parameters, x_train, network.activation.function)

		// cost function
		cost := costFunction(y_hat, y_train)

		// backward propagation
		dW, db := backwardPropagation(network.parameters, Z, A, y_train, y_hat, network.activation.derivative)

		// update parameters (gradient descent)
		network.parameters = updateParameters(network.parameters, dW, db, network.learning_rate)

		// print the cost every 1000 iterations
		if print_cost && i%1000 == 0 {
			fmt.Printf("it %d: | t: %.2fs | cost: %f \n", i, time.Since(start).Seconds(), cost)
			costs = append(costs, cost)
		}
	}

	return costs
}

// predictions
func (network *neuralNetwork) Predict(x *mat.Dense) *mat.Dense {
	// forward propagation
	predictions, _, _ := forwardPropagation(network.parameters, x, network.activation.function)

	return predictions
}
