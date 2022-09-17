const modelSelect = document.querySelector("#model");
const lengthInput = document.querySelector("#length");
const prefixInput = document.querySelector("#prefix");
const generateButton = document.querySelector("#generate");
const statusParagraph = document.querySelector("#status");
const midiPlayer = document.querySelector("#midi");
const downloadAnchor = document.querySelector("#download");

generateButton.addEventListener("click", async (e) => {
  e.preventDefault();
  generateButton.disabled = true;
  statusParagraph.innerText = "Loading...";
  midiPlayer.setAttribute("src", "data:,");
  downloadAnchor.setAttribute("href", "#");
  let url = "http://localhost/generate";
  let body = {
    method: "POST",
    mode: "cors",
    headers: {
      "Content-Type": "application/json",
      Accept: "audio/midi",
    },
    body: JSON.stringify({
      model: modelSelect.value,
      length: parseInt(lengthInput.value),
      prefix: prefixInput.value.split(",").map((elm) => parseInt(elm)),
    }),
  };
  console.log(url, body);
  try {
    let response = await fetch(url, body);
    if (response.ok) {
      let buffer = await response.arrayBuffer();
      let blob = new Blob([buffer], { type: "audio/midi" });
      let blobUrl = URL.createObjectURL(blob);
      statusParagraph.innerText = "Success!";
      midiPlayer.setAttribute("src", blobUrl);
      downloadAnchor.setAttribute("href", blobUrl);
    } else {
      statusParagraph.innerText = await response.text();
    }
  } catch (err) {
    statusParagraph.innerText = err;
  }
  generateButton.disabled = false;
});
