package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/pkg/errors"
	"tinygo.org/x/bluetooth"
)

const (
	deviceName = "Holy-IOT"
	filename   = "new/3.4m_2.txt"
	timeout    = 35 * time.Minute
)

var adapter = bluetooth.DefaultAdapter

func main() {
	if err := adapter.Enable(); err != nil {
		panic(errors.Wrap(err, "enable BLE stack"))
	}

	measurements := make(chan int16, 2)

	go func() {
		err := adapter.Scan(func(adapter *bluetooth.Adapter, device bluetooth.ScanResult) {
			if device.LocalName() == deviceName {
				measurements <- device.RSSI
			}
		})
		if err != nil {
			panic(errors.Wrap(err, "scan"))
		}
	}()

	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY, os.ModePerm)
	if err != nil {
		panic(errors.Wrap(err, "open output"))
	}

	defer file.Close()

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	for {
		select {
		case rssi := <-measurements:
			fmt.Fprintln(file, rssi)
		case <-ctx.Done():
			return
		case <-quit:
			return
		}
	}
}
