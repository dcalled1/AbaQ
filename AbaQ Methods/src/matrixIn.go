package main

import (
	"gonum.org/v1/gonum/mat"
)

var coeff *mat.Dense
var terms *mat.VecDense
var n int

func SetCoefficientMatrix(nVar int, data []float64) {
	n = nVar
	coeff = mat.NewDense(n, n, data)
}

func SetIndependentTerms(data []float64) {
	terms = mat.NewVecDense(len(data), data)
}

func getN() int {
	return n
}
