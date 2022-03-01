package neuralNetwork

import (
	"fmt"
	"math"
	"strconv"

	ngo "deep_learning/numeric"

	"gonum.org/v1/gonum/mat"
)

// initializing the model parameters
func InitializeParameters(nn_structure []int) map[string]*mat.Dense {
	parameters := make(map[string]*mat.Dense) // map containing the parameters
	L := len(nn_structure) - 1                // number of layers

	for l := 0; l < L; l++ {
		scalar := math.Sqrt((6.0 / float64(nn_structure[l]+nn_structure[l+1])))
		var init mat.Dense
		init.Scale(scalar, ngo.Randn(nn_structure[l+1], nn_structure[l]))
		parameters["W"+strconv.Itoa(l+1)] = &init
		parameters["b"+strconv.Itoa(l+1)] = mat.NewDense(nn_structure[l+1], 1, nil)
	}

	return parameters
}

// forward propagation step
func ForwardPropagation(parameters map[string]*mat.Dense, x *mat.Dense) (*mat.Dense, map[string]*mat.Dense, map[string]*mat.Dense) {
	L := len(parameters) / 2         // number of layers
	Z := make(map[string]*mat.Dense) // linear function
	A := make(map[string]*mat.Dense) // activation function
	A[strconv.Itoa(0)] = x

	for l := 0; l < L; l++ {
		W := parameters["W"+strconv.Itoa(l+1)] // weights W
		b := parameters["b"+strconv.Itoa(l+1)] // biases b

		var matmul mat.Dense
		matmul.Mul(W, A[strconv.Itoa(l)])

		Z[strconv.Itoa(l+1)] = ngo.AddMatrixVector(matmul, b) // compute the linear operation
		A[strconv.Itoa(l+1)] = ngo.Tanh(Z[strconv.Itoa(l+1)]) // compute the non linear operation
	}

	// prediction
	y_hat := Z[strconv.Itoa(L)]

	return y_hat, Z, A
}

// computing the cost function
func CostFunction(y_hat, y *mat.Dense) float64 {
	m := y_hat.RawMatrix().Cols

	var aux_1 mat.Dense
	aux_1.Sub(y_hat, y)
	aux_1.MulElem(&aux_1, &aux_1)
	sum := mat.Sum(&aux_1)

	return (1.0 / (2.0 * float64(m)) * sum)
}

// backward propagation step
func BackwardPropagation(parameters, Z, A map[string]*mat.Dense, y, y_hat *mat.Dense) (map[string]*mat.Dense, map[string]*mat.Dense) {
	m := y.RawMatrix().Cols  // number of training examples
	L := len(parameters) / 2 // number of layers

	dZ := make(map[string]*mat.Dense) // derivatives of the linear function Z
	dW := make(map[string]*mat.Dense) // derivatives of the weigths W
	db := make(map[string]*mat.Dense) // derivatives of the biases b
	dA := make(map[string]*mat.Dense) // derivatives of the activation function A

	var aux_1 mat.Dense
	aux_1.Sub(y_hat, y)
	aux_1.Scale(1./float64(m), &aux_1)
	dZ[strconv.Itoa(L)] = &aux_1

	var aux_2 mat.Dense
	aux_2.Mul(dZ[strconv.Itoa(L)], A[strconv.Itoa(L-1)].T())
	dW[strconv.Itoa(L)] = &aux_2

	db[strconv.Itoa(L)] = ngo.SumRows(dZ[strconv.Itoa(L)])

	for l := L - 1; l > 0; l-- {
		var aux_3 mat.Dense
		aux_3.Mul(parameters["W"+strconv.Itoa(l+1)].T(), dZ[strconv.Itoa(l+1)])
		dA[strconv.Itoa(l)] = &aux_3

		var aux_4 mat.Dense
		aux_4.MulElem(ngo.Tanh(Z[strconv.Itoa(l)]), ngo.Tanh(Z[strconv.Itoa(l)]))
		n, m := aux_4.Dims()
		aux_4.Sub(mat.NewDense(n, m, ngo.Linspace(1, 1, n*m)), &aux_4)
		aux_4.MulElem(dA[strconv.Itoa(l)], &aux_4)
		dZ[strconv.Itoa(l)] = &aux_4

		var aux_5 mat.Dense
		aux_5.Mul(dZ[strconv.Itoa(l)], A[strconv.Itoa(l-1)].T())
		dW[strconv.Itoa(l)] = &aux_5

		db[strconv.Itoa(l)] = ngo.SumRows(dZ[strconv.Itoa(l)])
	}

	return dW, db
}

// update the parameters
func UpdateParameters(parameters, dW, db map[string]*mat.Dense, learning_rate float64) map[string]*mat.Dense {
	L := len(parameters) / 2 // number of layers

	for l := 0; l < L; l++ {
		var aux_1 mat.Dense
		aux_1.Scale(learning_rate, dW[strconv.Itoa(l+1)])
		aux_1.Sub(parameters["W"+strconv.Itoa(l+1)], &aux_1)
		parameters["W"+strconv.Itoa(l+1)] = &aux_1

		var aux_2 mat.Dense
		aux_2.Scale(learning_rate, db[strconv.Itoa(l+1)])
		aux_2.Sub(parameters["b"+strconv.Itoa(l+1)], &aux_2)
		parameters["b"+strconv.Itoa(l+1)] = &aux_2
	}

	return parameters
}

// train model
func Fit(x_train, y_train *mat.Dense, nn_structure []int, num_iterations int, learning_rate float64, print_cost bool) (map[string]*mat.Dense, []float64) {
	// keep track of the cost
	costs := []float64{}

	// initializing the model parameters
	parameters := InitializeParameters(nn_structure)

	// loop
	for i := 0; i < num_iterations; i++ {
		// forward propagation
		y_hat, Z, A := ForwardPropagation(parameters, x_train)

		// cost function
		cost := CostFunction(y_hat, y_train)

		// backward propagation
		dW, db := BackwardPropagation(parameters, Z, A, y_train, y_hat)

		// update parameters (Gradient descent)
		parameters = UpdateParameters(parameters, dW, db, learning_rate)

		// print the cost every 1000 iterations
		if print_cost && i%1000 == 0 {
			fmt.Printf("cost after iteration %d: %f \n", i, cost)
			costs = append(costs, cost)
		}
	}

	return parameters, costs
}

// predictions
func Predict(parameters map[string]*mat.Dense, x *mat.Dense) *mat.Dense {
	// forward propagation
	predictions, _, _ := ForwardPropagation(parameters, x)

	return predictions
}
