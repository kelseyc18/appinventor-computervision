const submitButton = document.getElementById('submitButton');
const imageData = document.getElementById('imageData');
const imageEl = document.getElementById('sampleImage');

const math = new dl.NDArrayMathGPU();
// squeezenet is loaded from https://unpkg.com/deeplearn-squeezenet
const squeezeNet = new squeezenet.SqueezeNet(math);

async function lol() {
  await squeezeNet.load();

  console.log("DeepLearnJS: deeplearn-main.js load");
  DeepLearnJS.ready();
}

console.log("DeepLearnJS: deeplearn-main.js start");
lol();

async function infer(imageData) {
  console.log("DeepLearnJS: in infer");
  var img = new Image(227, 227);

  img.onload = async function() {
    console.log("DeepLearnJS: onload");
    const image = dl.Array3D.fromPixels(img);
    console.log("DeepLearnJS: fromPixels");
    const inferenceResult = await squeezeNet.predict(image);
    console.log("DeepLearnJS: squeezeNet predict");
    await inferenceResult.logits.data();

    const topClassesToProbs = await squeezeNet.getTopKClasses(inferenceResult.logits, 10);
    
    console.log("DeepLearnJS: squeezeNet getTopK");

    var result = {};

    for (const className in topClassesToProbs) {
      result[className] = topClassesToProbs[className].toFixed(5);
    }
    
    console.log("DeepLearnJS: JSON stringify is " + JSON.stringify(result));

    DeepLearnJS.reportResult(JSON.stringify(result));
    
    console.log("DeepLearnJS: reportResult called");
  }

  img.src = 'data:image/png;base64,' + imageData;
  sampleImage.src = img.src;
  console.log("DeepLearnJS: end infer");
  return 'ok';
}

function onSubmit() {
  infer(imageData.value);
}
