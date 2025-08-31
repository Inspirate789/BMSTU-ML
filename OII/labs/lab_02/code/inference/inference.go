package inference

import (
	"github.com/sbiemont/fugologic/builder"
	"github.com/sbiemont/fugologic/crisp"
	"github.com/sbiemont/fugologic/fuzzy"
	"github.com/sbiemont/fugologic/id"
)

const eps = 1e-5

var (
	fvSpeed, fvDistance, fvNewSpeed *fuzzy.IDVal
	engine                          fuzzy.Engine
)

func init() {
	crispSpeed, _ := crisp.NewSetN(0, 180, 1/eps)
	fsSpeed, _ := fuzzy.NewIDSets(map[id.ID]fuzzy.SetBuilder{
		"Very low":  fuzzy.StepDown{A: 1, B: 7},
		"Low":       fuzzy.Trapezoid{A: 5, B: 10, C: 20, D: 30},
		"Medium":    fuzzy.Trapezoid{A: 25, B: 30, C: 65, D: 75},
		"High":      fuzzy.Trapezoid{A: 70, B: 80, C: 115, D: 125},
		"Very high": fuzzy.StepUp{A: 110, B: 130},
	})
	fvSpeed, _ = fuzzy.NewIDVal("Speed", crispSpeed, fsSpeed)

	crispDistance, _ := crisp.NewSetN(0, 100, 1/eps)
	fsDistance, _ := fuzzy.NewIDSets(map[id.ID]fuzzy.SetBuilder{
		"Very low":  fuzzy.StepDown{A: 1, B: 5},
		"Low":       fuzzy.Trapezoid{A: 3, B: 7, C: 9, D: 15},
		"Medium":    fuzzy.Triangular{A: 9, B: 15, C: 24},
		"High":      fuzzy.Trapezoid{A: 15, B: 24, C: 50, D: 60},
		"Very high": fuzzy.StepUp{A: 45, B: 60},
	})
	fvDistance, _ = fuzzy.NewIDVal("Distance", crispDistance, fsDistance)

	crispNewSpeed, _ := crisp.NewSetN(0, 180, 1/eps)
	fsNewSpeed, _ := fuzzy.NewIDSets(map[id.ID]fuzzy.SetBuilder{
		"Very low":  fuzzy.StepDown{A: 1, B: 7},
		"Low":       fuzzy.Trapezoid{A: 5, B: 10, C: 20, D: 30},
		"Medium":    fuzzy.Trapezoid{A: 25, B: 30, C: 65, D: 75},
		"High":      fuzzy.Trapezoid{A: 70, B: 80, C: 115, D: 125},
		"Very high": fuzzy.StepUp{A: 110, B: 130},
	})
	fvNewSpeed, _ = fuzzy.NewIDVal("New speed", crispNewSpeed, fsNewSpeed)

	bld := builder.Config{
		Optr:   fuzzy.OperatorZadeh{},
		Impl:   fuzzy.ImplicationProd,
		Agg:    fuzzy.AggregationUnion,
		Defuzz: fuzzy.DefuzzificationCentroid,
	}.FuzzyAssoMatrix()
	_ = bld.
		Asso(fvSpeed, fvDistance, fvNewSpeed).
		Matrix(
			[]id.ID{"Very low", "Low", "Medium", "High", "Very high"}, // Speed
			map[id.ID][]id.ID{ // Distance
				"Very low":  {"Very low", "Very low", "Very low", "Very low", "Very low"},
				"Low":       {"Very low", "Very low", "Low", "Low", "Low"},
				"Medium":    {"Very low", "Low", "Medium", "Medium", "High"},
				"High":      {"Low", "Medium", "High", "High", "Very high"},
				"Very high": {"Medium", "Medium", "High", "Very high", "Very high"},
			},
		)

	engine, _ = bld.Engine()
}

func Inference(speed, distance float64) float64 {
	res, _ := engine.Evaluate(fuzzy.DataInput{
		fvSpeed:    speed,
		fvDistance: distance,
	})

	return res[fvNewSpeed]
}
