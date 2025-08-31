package inference_test

import (
	"fmt"
	"lab_02/inference"
	"math"
	"testing"
)

func BenchmarkInference(b *testing.B) {
	for i := 0; i < b.N; i++ {
		res := inference.Inference(50, 15)
		_ = res
	}
}

func TestDeviation(t *testing.T) {
	const (
		eps             = 1e-5
		iterationsCount = 100
		targetDistance  = 15.
	)

	fmt.Printf("[")

	for leadingSpeed := 5.; leadingSpeed <= 130.; leadingSpeed++ {
		var (
			trailingSpeed, distance = leadingSpeed, targetDistance // km/h, meters
			variance                = 0.
		)

		//fmt.Printf("leader: %.3g km/h, trailer: %.3g km/h, distance: %.3g m, target: %.3g m\n",
		//	leadingSpeed, trailingSpeed, distance, targetDistance,
		//)

		for range iterationsCount {
			trailingSpeed = inference.Inference(trailingSpeed, distance)
			distance -= (trailingSpeed - leadingSpeed) / 3.6

			variance += (distance - targetDistance) * (distance - targetDistance)

			if distance < eps {
				panic("crash!")
			}
		}

		//fmt.Printf("standard deviation: %.3g m\n", math.Sqrt(variance/iterationsCount))
		fmt.Printf("%.3g, ", math.Sqrt(variance/iterationsCount))
	}

	fmt.Printf("]\n")
}
