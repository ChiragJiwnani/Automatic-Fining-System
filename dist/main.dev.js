"use strict";

var video = document.getElementById('videoElement');
var canvas = document.getElementById('canvasElement');
var ctx = canvas.getContext('2d');
navigator.mediaDevices.getUserMedia({
  video: true
}).then(function (stream) {
  video.srcObject = stream;
})["catch"](function (err) {
  console.log("An error occurred: " + err);
});
video.addEventListener('play', function () {
  var frameInterval = 1000 / 30; // Adjust for desired frame rate

  setInterval(function () {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height); // Send imageData to backend for processing
    // Update UI with detected license plate information
  }, frameInterval);
});
//# sourceMappingURL=main.dev.js.map
