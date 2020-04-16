package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Index struct {
	I, J int
}

type Reg1 struct {
	Mat *mat.Dense
	Mul map[Index]float64
}

type Reg2 struct {
	Mat   *mat.Dense
	Mul   map[Index]float64
	Marks []int
}

type Reg3 struct {
	L, U *mat.Dense
}

type Reg4 struct {
	X    []float64
	Disp float64
	It   int
}

var euclidean, relative, badIn bool

var gSimStages, gParStages []Reg1
var gTotStages []Reg2
var dooStages, cholStages, croutStages []Reg3
var jacTable, seidTable []Reg4

func RegressiveSubs(Ab *mat.Dense) (ans *mat.VecDense) {
	x := make([]float64, n)
	x[n-1] = Ab.At(n-1, n) / Ab.At(n-1, n-1)
	for i := n - 2; i >= 0; i-- {
		sum := float64(0)
		for p := i + 1; p <= n-1; p++ {
			sum = sum + Ab.At(i, p)*x[p]
		}
		x[i] = (Ab.At(i, n) - sum) / Ab.At(i, i)
	}
	ans = mat.NewVecDense(n, x)
	return
}

func ProgressiveSubs(Ab *mat.Dense) (ans *mat.VecDense) {
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := float64(0)
		for p := 0; p > i; p++ {
			sum = sum + Ab.At(i, p)*x[p]
		}
		x[n-i-1] = (Ab.At(i, n) - sum) / Ab.At(i, i)
	}
	ans = mat.NewVecDense(n, x)
	return
}

func GaussElimination() *mat.Dense {
	gSimStages = make([]Reg1, 0, n)
	Ab := mat.NewDense(n, n+1, nil)
	Ab.Augment(coeff, terms)
	gSimStages = append(gSimStages, Reg1{mat.DenseCopyOf(Ab), make(map[Index]float64, 0)})
	for k := 0; k < n-1; k++ {
		for rnd := k + 1; Ab.At(k, k) == 0 && rnd < n-1; rnd++ {
			swapRows(Ab, k, rnd)
		}
		mults := make(map[Index]float64)
		for i := k + 1; i < n; i++ {
			mult := Ab.At(i, k) / Ab.At(k, k)
			mults[Index{i, k}] = mult
			for j := k; j < n+1; j++ {
				Ab.Set(i, j, Ab.At(i, j)-mult*Ab.At(k, j))
			}
		}
		gSimStages = append(gSimStages, Reg1{mat.DenseCopyOf(Ab), mults})
	}
	return Ab
}

func Gauss() (ans *mat.VecDense) {
	ans = RegressiveSubs(GaussElimination())
	return
}

func swapRows(m *mat.Dense, i, j int) {
	_, nCol := m.Caps()
	for k := 0; k < nCol; k++ {
		a, b := m.At(i, k), m.At(j, k)
		m.Set(j, k, a)
		m.Set(i, k, b)
	}

}

func swapCols(m *mat.Dense, i, j int) {
	nRow, _ := m.Caps()
	for k := 0; k < nRow; k++ {
		a, b := m.At(k, i), m.At(k, j)
		m.Set(k, j, a)
		m.Set(k, i, b)
	}
}

func PartialPivoting(Ab *mat.Dense, k int) {
	max := math.Abs(Ab.At(k, k))
	topRow := k
	for s := k + 1; s < n; s++ {
		if newMax := math.Abs(Ab.At(s, k)); newMax > max {
			max = newMax
			topRow = s
		}
	}
	if max == 0 {
		badIn = true
		return
	}
	if topRow != k {
		swapRows(Ab, topRow, k)
	}

}

func GaussEliminationPartialPivoting() *mat.Dense {
	gParStages = make([]Reg1, 0, n)
	Ab := mat.NewDense(n, n+1, nil)
	Ab.Augment(coeff, terms)
	gParStages = append(gParStages, Reg1{mat.DenseCopyOf(Ab), make(map[Index]float64, 0)})
	for k := 0; k < n-1; k++ {
		PartialPivoting(Ab, k)
		mults := make(map[Index]float64)
		for i := k + 1; i < n; i++ {
			mult := Ab.At(i, k) / Ab.At(k, k)
			mults[Index{i, k}] = mult
			for j := k; j < n+1; j++ {
				Ab.Set(i, j, Ab.At(i, j)-mult*Ab.At(k, j))
			}
		}
		gParStages = append(gParStages, Reg1{mat.DenseCopyOf(Ab), mults})
	}
	return Ab
}

func GaussPartialPivoting() (ans *mat.VecDense) {
	ans = RegressiveSubs(GaussEliminationPartialPivoting())
	return
}

func TotalPivoting(Ab *mat.Dense, k int, marks []int) {
	max := float64(0)
	topRow := k
	topCol := k
	for r := k; r < n; r++ {
		for s := k; s < n; s++ {
			if newMax := math.Abs(Ab.At(r, s)); newMax > max {
				max = newMax
				topRow = r
				topCol = s
			}
		}
	}
	if max == 0 {
		badIn = true
		return
	}
	if topRow != k {
		swapRows(Ab, topRow, k)
	}
	if topCol != k {
		swapCols(Ab, topCol, k)
		marks[topCol], marks[k] = marks[k], marks[topCol]
	}
}

func GaussEliminationTotalPivoting() (*mat.Dense, []int) {
	gTotStages = make([]Reg2, 0, n)
	Ab := mat.NewDense(n, n+1, nil)
	Ab.Augment(coeff, terms)
	marks := make([]int, n)
	for index := 0; index < n; index++ {
		marks[index] = index
	}
	markscopy := make([]int, n)
	copy(markscopy, marks)
	gTotStages = append(gTotStages, Reg2{mat.DenseCopyOf(Ab), make(map[Index]float64, 0), markscopy})
	for k := 0; k < n-1; k++ {
		TotalPivoting(Ab, k, marks)
		mults := make(map[Index]float64)
		markscopy = make([]int, n)
		copy(markscopy, marks)
		for i := k + 1; i < n; i++ {
			mult := Ab.At(i, k) / Ab.At(k, k)
			mults[Index{i, k}] = mult
			for j := k; j < n+1; j++ {
				Ab.Set(i, j, Ab.At(i, j)-mult*Ab.At(k, j))
			}
		}
		gTotStages = append(gTotStages, Reg2{mat.DenseCopyOf(Ab), mults, markscopy})
	}
	return Ab, marks
}

func GaussTotalPivoting() (ans *mat.VecDense) {
	u, marks := GaussEliminationTotalPivoting()
	anstemp := RegressiveSubs(u)
	data := make([]float64, n)
	for i, e := range marks {
		data[i] = anstemp.AtVec(e)
	}
	ans = mat.NewVecDense(n, data)
	return
}

func DirectFactorization(method int) (*mat.Dense, *mat.Dense) {
	L := mat.NewDense(n, n, nil)
	U := mat.NewDense(n, n, nil)

	switch method {
	case 1:
		dooStages = make([]Reg3, 0, n)
	case 2:
		croutStages = make([]Reg3, 0, n)
	case 3:
		cholStages = make([]Reg3, 0, n)
	}

	for k := 0; k < n; k++ {
		sum1 := float64(0)
		for p := 0; p < k; p++ {
			sum1 += L.At(k, p) * U.At(p, k)
		}
		val := coeff.At(k, k) - sum1
		switch method {
		case 1:
			U.Set(k, k, val)
			L.Set(k, k, 1)
		case 2:
			U.Set(k, k, 1)
			L.Set(k, k, val)
		case 3:
			newVal := math.Sqrt(val)
			U.Set(k, k, newVal)
			L.Set(k, k, newVal)
		}

		for i := k + 1; i < n; i++ {
			sum2 := float64(0)
			for p := 0; p < k; p++ {
				sum2 += L.At(i, p) * U.At(p, k)
			}
			L.Set(i, k, (coeff.At(i, k)-sum2)/U.At(k, k))
		}

		for j := k + 1; j < n; j++ {
			sum3 := float64(0)
			for p := 0; p < k; p++ {
				sum3 += L.At(k, p) * U.At(p, j)
			}
			U.Set(k, j, (coeff.At(k, j)-sum3)/L.At(k, k))
		}
		switch method {
		case 1:
			dooStages = append(dooStages, Reg3{mat.DenseCopyOf(L), mat.DenseCopyOf(U)})
		case 2:
			croutStages = append(croutStages, Reg3{mat.DenseCopyOf(L), mat.DenseCopyOf(U)})
		case 3:
			cholStages = append(cholStages, Reg3{mat.DenseCopyOf(L), mat.DenseCopyOf(U)})
		}
	}

	return L, U
}

func DoolittleFactorization() *mat.VecDense {
	L, U := DirectFactorization(1)
	Lb := mat.NewDense(n, n+1, nil)
	Lb.Augment(L, terms)
	z := ProgressiveSubs(Lb)
	Uz := mat.NewDense(n, n+1, nil)
	Uz.Augment(U, z)
	x := RegressiveSubs(Uz)
	return x

}

func CroutFactorization() *mat.VecDense {
	L, U := DirectFactorization(2)
	Lb := mat.NewDense(n, n+1, nil)
	Lb.Augment(L, terms)
	z := ProgressiveSubs(Lb)
	Uz := mat.NewDense(n, n+1, nil)
	Uz.Augment(U, z)
	x := RegressiveSubs(Uz)
	return x

}

func CholeskyFactorization() *mat.VecDense {
	L, U := DirectFactorization(3)
	Lb := mat.NewDense(n, n+1, nil)
	Lb.Augment(L, terms)
	z := ProgressiveSubs(Lb)
	Uz := mat.NewDense(n, n+1, nil)
	Uz.Augment(U, z)
	x := RegressiveSubs(Uz)
	return x

}

func Jacobi(x []float64, tol float64, maxIt uint) ([]float64, uint, bool) {
	jacTable = make([]Reg4, 0, maxIt)
	x0 := make([]float64, n)
	x0cop := make([]float64, n)
	copy(x0, x)
	count := uint(0)
	disp := math.MaxFloat64
	jacTable = append(jacTable, Reg4{x0cop, math.Inf(1), 0})
	var x1 []float64
	for disp > tol && count < maxIt {
		x1 = newJacobiSet(x0)
		copy(x0cop, x1)
		disp = norm(x1, x0)
		jacTable = append(jacTable, Reg4{x0cop, disp, int(count + 1)})
		x0 = x1
		count++
	}

	if disp < tol {
		return x1, count, true
	}
	return x1, count, false
}

func newJacobiSet(xl []float64) []float64 {
	xn := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := float64(0)
		for j := 0; j < n; j++ {
			if i != j {
				sum += coeff.At(i, j) * xl[j]
			}
		}
		xn[i] = (terms.AtVec(i) - sum) / coeff.At(i, i)
	}
	return xn
}

func GaussSeidelRelaxed(x []float64, w, tol float64, maxIt uint) ([]float64, uint, bool) {
	seidTable = make([]Reg4, 0, maxIt)
	x0 := make([]float64, n)
	copy(x0, x)
	x0cop := make([]float64, n)
	copy(x0cop, x0)
	count := uint(0)
	disp := math.MaxFloat64
	seidTable = append(seidTable, Reg4{x0cop, math.Inf(1), 0})
	var x1 []float64
	for disp > tol && count < maxIt {
		x1 = newGaussSet(x0, w)
		x0cop = make([]float64, n)
		copy(x0cop, x1)
		disp = norm(x1, x0)
		seidTable = append(seidTable, Reg4{x0cop, disp, int(count + 1)})
		x0 = x1
		count++
	}

	if disp < tol {
		return x1, count, true
	}
	return x1, count, false
}

func newGaussSet(xl []float64, w float64) []float64 {
	xn := make([]float64, n)
	for i := 0; i < n; i++ {
		sum1, sum2 := float64(0), float64(0)
		for j := 0; j < i; j++ {
			sum2 += coeff.At(i, j) * xn[j]
		}
		for j := i + 1; j < n; j++ {
			sum1 += coeff.At(i, j) * xl[j]
		}
		if coeff.At(i, i) == 0 {
			badIn = true
		}
		sum := sum1 + sum2
		xn[i] = (1-w)*xl[i] - w*(sum+terms.AtVec(i))/coeff.At(i, i)
		//xn[I] = (terms.AtVec(I) - sum) / coeff.At(I, I)
	}

	return xn
}

func norm(x, xl []float64) float64 {
	ans := float64(0)
	if euclidean {
		sum := float64(0)
		for i := 0; i < n; i++ {
			if relative && x[i] != 0 {
				sum += math.Pow((x[i]-xl[i])/x[i], 2)
			} else {
				sum += math.Pow(x[i]-xl[i], 2)
			}
		}
		ans = sum
	} else {
		max := math.Inf(-1)
		for i := 0; i < n; i++ {
			if relative && x[i] != 0 {
				if newMax := math.Abs((x[i] - xl[i]) / x[i]); newMax > max {
					max = newMax
				}
			} else {
				if newMax := math.Abs(x[i] - xl[i]); newMax > max {
					max = newMax
				}
			}
		}
		ans = max
	}

	return ans
}

func SetRelative() {
	relative = true
}

func SetAbsulute() {
	relative = false
}

func SetInfinity() {
	euclidean = false
}

func SetEuclidean() {
	euclidean = true
}

func GetGaussSimpleStages() []Reg1 {
	return gSimStages
}

func GetGaussPartialStages() []Reg1 {
	return gParStages
}

func GetGaussTotalStages() []Reg2 {
	return gTotStages
}

func GetDoolittleStages() []Reg3 {
	return dooStages
}

func GetCroutStages() []Reg3 {
	return croutStages
}

func GetCholeskyStages() []Reg3 {
	return cholStages
}

func GetJacobiTable() []Reg4 {
	return jacTable
}

func GetGaussSeidelTable() []Reg4 {
	return seidTable
}

func BadIn() bool {
	return badIn
}
