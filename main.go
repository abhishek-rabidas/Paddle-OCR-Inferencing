package main

import (
	"fmt"
	"gocv.io/x/gocv"
	"image"
	"image/color"
)

type PaddleOCR struct {
	net         *gocv.Net
	outputNames []string
	window1     *gocv.Window
	window2     *gocv.Window
	image       gocv.Mat
}

func main() {
	var pd = PaddleOCR{}
	pd.Load()
	pd.loadImage("./asset/card.jpg")
	frame := pd.detector()
	if frame.Empty() {
		fmt.Println("EMpty")
		return
	}
	pd.window1.IMShow(pd.image)
	pd.window2.IMShow(frame)
	pd.window1.WaitKey(0)
	pd.window2.WaitKey(0)
}

func (pd *PaddleOCR) Load() {
	net := gocv.ReadNetFromONNX("./Models/det.onnx")

	net.SetPreferableTarget(gocv.NetTargetCPU)
	net.SetPreferableBackend(gocv.NetBackendDefault)

	pd.net = &net
	pd.outputNames = getOutputsNames(&net)
	pd.window1 = gocv.NewWindow("Original")
	pd.window2 = gocv.NewWindow("Paddle")

}

func (pd *PaddleOCR) loadImage(img string) {
	mat := gocv.IMRead(img, gocv.IMReadColor)
	pd.image = mat

}

func (pd *PaddleOCR) detector() gocv.Mat {
	img := pd.image.Clone()
	img.ConvertTo(&img, gocv.MatTypeCV32F)
	blob := gocv.BlobFromImage(img, 1.0, image.Pt(416, 416), gocv.NewScalar(0, 0, 0, 0), true, false)
	pd.net.SetInput(blob, "")
	probs := pd.net.Forward(pd.outputNames[0])
	fmt.Println(probs)
	return img
	/*	boxes, confidences := postProcess(img, &probs)
		indices := make([]int, 100)
		if len(boxes) == 0 { // No Classes
			panic("empty")
		}
		gocv.NMSBoxes(boxes, confidences, 0.45, 0.5, indices)
		fmt.Println(confidences)*/
	//return drawRect(pd.image, boxes, indices)

}

func postProcess(frame gocv.Mat, out *gocv.Mat) ([]image.Rectangle, []float32) {
	var confidences []float32
	var boxes []image.Rectangle

	data, _ := out.DataPtrFloat32()
	for i := 0; i < out.Rows(); i, data = i+1, data[out.Cols():] {

		scoresCol := out.RowRange(i, i+1)

		scores := scoresCol.ColRange(5, out.Cols())
		_, confidence, _, _ := gocv.MinMaxLoc(scores)
		if confidence > 0.5 {

			centerX := int(data[0] * float32(frame.Cols()))
			centerY := int(data[1] * float32(frame.Rows()))
			width := int(data[2] * float32(frame.Cols()))
			height := int(data[3] * float32(frame.Rows()))

			left := centerX - width/2
			top := centerY - height/2
			confidences = append(confidences, float32(confidence))
			boxes = append(boxes, image.Rect(left, top, width, height))
		}
	}

	return boxes, confidences
}

func drawRect(img gocv.Mat, boxes []image.Rectangle, indices []int) gocv.Mat {
	for _, idx := range indices {
		gocv.Rectangle(&img, image.Rect(boxes[idx].Max.X, boxes[idx].Max.Y, boxes[idx].Max.X+boxes[idx].Min.X, boxes[idx].Max.Y+boxes[idx].Min.Y), color.RGBA{255, 0, 0, 0}, 2)
	}
	return img
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
