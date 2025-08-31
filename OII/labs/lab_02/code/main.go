package main

import (
	"fmt"
	"lab_02/inference"
	"math"
)

func main() {
	const (
		eps             = 1e-5
		leadingSpeed    = 50.
		iterationsCount = 100
		targetDistance  = 15.
	)
	var (
		trailingSpeed, distance = leadingSpeed, targetDistance // km/h, meters
		variance                = 0.
	)

	fmt.Printf("leader: %.3g km/h, trailer: %.3g km/h, distance: %.3g m, target: %.3g m\n",
		leadingSpeed, trailingSpeed, distance, targetDistance,
	)

	for range iterationsCount {
		trailingSpeed = inference.Inference(trailingSpeed, distance)
		distance -= (trailingSpeed - leadingSpeed) / 3.6
		// fmt.Printf("leader: %.3g km/h, trailer: %.3g km/h, distance: %.3g m\n", leadingSpeed, trailingSpeed, distance)

		variance += (distance - targetDistance) * (distance - targetDistance)

		if distance < eps {
			panic("crash!")
		}
	}

	fmt.Printf("standard deviation: %.3g m\n", math.Sqrt(variance/iterationsCount))
}
