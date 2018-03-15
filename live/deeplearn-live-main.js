const submitButton = document.getElementById('submitButton');

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

var video = document.createElement('video');
video.setAttribute('autoplay', '');
video.setAttribute('playsinline', '');

document.body.appendChild(video);

// Setup webcam
navigator.mediaDevices.getUserMedia({video: true, audio: false})
.then((stream) => {
  video.srcObject = stream;
  video.width = 227;
  video.height = 227;
}).catch((err) => {
  console.log("error");
  console.log(err);
});

async function infer() {
  console.log("DeepLearnJS: in infer");

  console.log("DeepLearnJS: onload");
  const image = dl.Array3D.fromPixels(video);
  console.log("DeepLearnJS: fromPixels");
  const logits = squeezeNet.predict(image);
  console.log("DeepLearnJS: squeezeNet predict");

  const topClassesToProbs = await squeezeNet.getTopKClasses(logits, 10);
  
  console.log("DeepLearnJS: squeezeNet getTopK");

  var result = [];

  for (const className in topClassesToProbs) {
    result.push([className, topClassesToProbs[className].toFixed(5)]);
  }
    
  console.log("DeepLearnJS: JSON stringify is " + JSON.stringify(result));

  DeepLearnJS.reportResult(JSON.stringify(result));
    
  console.log("DeepLearnJS: reportResult called");

  console.log("DeepLearnJS: end infer");
  return 'ok';
}

function onSubmit() {
  infer();
}
