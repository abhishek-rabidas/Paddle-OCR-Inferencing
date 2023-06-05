package main

import "gocv.io/x/gocv"

type PaddleOCR struct {
	net         *gocv.Net
	outputNames []string
	window      *gocv.Window
	image       gocv.Mat
}

func main() {
	var pd PaddleOCR = PaddleOCR{}
	pd.Load()
	pd.loadImage("./asset/test.jpg")
}

func (pd *PaddleOCR) Load() {
	net := gocv.ReadNetFromONNX("./Models/det.onnx")

	net.SetPreferableTarget(gocv.NetTargetCPU)
	net.SetPreferableBackend(gocv.NetBackendDefault)

	pd.net = &net
	pd.outputNames = getOutputsNames(&net)
	pd.window = gocv.NewWindow("Paddle OCR")

}

func (pd *PaddleOCR) loadImage(img string) {
	mat := gocv.IMRead(img, gocv.IMReadAnyColor)
	pd.image = mat
	pd.window.IMShow(pd.image)
}

func (pd *PaddleOCR) detector() {

}

func getOutputsNames(net *gocv.Net) []string {
	var outputLayers []string
	for _, i := range net.GetUnconnectedOutLayers() {
		layer := net.GetLayer(i)
		layerName := layer.GetName()
		if layerName != "_input" {
			outputLayers = append(outputLayers, layerName)
		}
	}
	return outputLayers
}
